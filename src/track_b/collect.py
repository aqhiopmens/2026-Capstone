"""안건 12 통제 수집(사전 고정 문서 docs/track_b_collect_prereg.md 개정 3a 구현): 서버 3개, 러너(오케스트레이터), 캡처, 매칭.

라벨은 러너 로그의 (client_ip, client_port, server_ip, server_port) + 시각에서만 가져온다. IP·서버로 라벨을 정하지 않는다.
자기 서버·자기 LAN에만 쓴다(외부 대상 없음).

  certs   인증서 생성 (S1 RSA-2048 단일, S2 ECDSA P-256, S3 RSA-4096 2단 체인)
  serve   서버 3개 구동 + 연결 수락 로그(독립 기준). 기본 127.0.0.1, LAN 수집 때만 --bind 0.0.0.0
  run     오케스트레이터: 학습 단계 S1~S3 → 시험 단계 S1~S3, 라운드마다 칸(정상·공격 × requests·curl)당 10연결 쿼터
  capture dumpcap 래퍼 (classic pcap)
  match   pcap 플로우 + 러너 로그 + 서버 수락 로그 -> 라벨 붙은 flows.csv, 쿼터 검사, 칸별 보고
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import ssl
import statistics
import string
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVERS = {"S1": 8443, "S2": 8444, "S3": 8445}
USER, PASSWORD = "admin", "Tz9k-Qm4Xb"  # 정답 암호는 10자(CORRECT_LEN)
CORRECT_LEN = len(PASSWORD)
DUMPCAP = r"C:\Program Files\Wireshark\dumpcap.exe"
PAGE = b"<html><body><h1>Portal</h1><form method=post action=/login>user <input name=u> pass <input name=p type=password></form></body></html>"
ALNUM = string.ascii_letters + string.digits

# ---------- 사전 고정: 라운드 칸당 10연결 배정표 (문서 1절) ----------
GET_FIRST_M, POST_FIRST_M = [2, 3, 4, 6], [2, 3, 4, 5]  # m≥2 첫 요청 GET 50% / POST 50%, m 다중집합 {1,1,2,2,3,3,4,4,5,6}
CELL_SPECS = [("GET", 1), ("POST", 1)] + [("GET", m) for m in GET_FIRST_M] + [("POST", m) for m in POST_FIRST_M]
LOGIN_CELLS = CELL_SPECS[1:]  # 정상 실패 전용 2/10은 GET 1(요청 하나뿐)을 뺀 9칸에 라운드마다 돌려 배정한다(고정하면 "그 칸에서 실패로 끝남 = 공격"이 된다)


def failonly_cells(rnd, impl):
    """라운드 rnd(0부터)·구현 impl에서 정상이 실패 전용으로 끝나는 칸 2개. 9라운드 한 바퀴에 칸마다 정확히 2번(구현당)."""
    k = 4 * (impl == "curl")
    return {LOGIN_CELLS[(2 * rnd + k + j) % len(LOGIN_CELLS)] for j in (0, 1)}
BROWSE_PER_CELL = 2  # 정상 칸 참고 부분집합(subset=browse), 주 지표 제외


def pw_len(m, i, last):
    """암호 길이 배정표: 마지막 요청(POST)은 10자(정상 성공 암호와 공격 오답이 같은 길이), 그 밖은 6~14를 (m, 위치)로 결정."""
    return CORRECT_LEN if last else 6 + ((3 * m + 5 * i) % 9)


def episode_plan(cls, first, m, failonly=False):
    """에피소드 하나의 요청 배정표. 반환 (ep_type, [dict(method, path, pw_len, expect_ok)]). 정상과 공격이 같은 (first, m)에서 같은 요청 종류·같은 길이를 갖는다.
    failonly: 정상이 끝까지 실패하는 에피소드(공격과 응답 결과가 같음)."""
    if m == 1 and first == "GET":
        return "none", [dict(method="GET", path="/login", pw_len=None, expect_ok=False)]
    ep_type = ("failonly" if failonly else "success") if cls == "normal" else "fail"
    reqs = [dict(method="GET", path="/login", pw_len=None, expect_ok=False)] if first == "GET" else []
    for _ in range(m - len(reqs)):
        i = len(reqs)
        last = i == m - 1
        reqs.append(dict(method="POST", path="/login", pw_len=pw_len(m, i, last), expect_ok=bool(last and ep_type == "success")))
    return ep_type, reqs


def browse_plan():
    return "browse", [dict(method="GET", path="/", pw_len=None, expect_ok=False), dict(method="GET", path="/login", pw_len=None, expect_ok=False)]


def password_for(req, rng):
    if req["method"] != "POST":
        return None
    if req["expect_ok"]:
        return PASSWORD
    while True:
        pw = "".join(rng.choice(ALNUM) for _ in range(req["pw_len"]))
        if pw != PASSWORD:
            return pw


def round_episodes(rng, rnd=0):
    """한 라운드(rnd: 0부터)의 모든 에피소드(칸당 10 + 정상 참고 2)를 시드 고정으로 섞는다. 항목: (cls, impl, first, m, subset, failonly)."""
    fo = {impl: failonly_cells(rnd, impl) for impl in ("requests", "curl")}
    eps = [(cls, impl, first, m, "main", cls == "normal" and (first, m) in fo[impl]) for cls in ("normal", "attack") for impl in ("requests", "curl") for first, m in CELL_SPECS]
    eps += [("normal", impl, 0, 0, "browse", False) for impl in ("requests", "curl") for _ in range(BROWSE_PER_CELL)]
    rng.shuffle(eps)
    return eps


# ---------- 인증서 ----------
def make_certs(out):
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    from cryptography.x509.oid import NameOID

    def name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    def issue(subject_cn, key, issuer_cn, issuer_key, ca=False, san=False):
        now = datetime.datetime.now(datetime.timezone.utc)
        builder = (x509.CertificateBuilder().subject_name(name(subject_cn)).issuer_name(name(issuer_cn))
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=90))
                   .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
        if san:
            builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        return builder.sign(issuer_key, hashes.SHA256())

    def pem_cert(cert):
        return cert.public_bytes(serialization.Encoding.PEM)

    def pem_key(key):
        return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())

    out.mkdir(parents=True, exist_ok=True)
    rsa2048 = rsa.generate_private_key(65537, 2048)
    s1 = issue("s1.lab", rsa2048, "s1.lab", rsa2048, san=True)
    ec_key = ec.generate_private_key(ec.SECP256R1())
    s2 = issue("s2.lab", ec_key, "s2.lab", ec_key, san=True)
    root_key, mid_key, leaf_key = (rsa.generate_private_key(65537, 4096) for _ in range(3))
    root = issue("root.lab", root_key, "root.lab", root_key, ca=True)
    mid = issue("mid.lab", mid_key, "root.lab", root_key, ca=True)
    s3 = issue("s3.lab", leaf_key, "mid.lab", mid_key, san=True)
    for tag, chain, key in (("S1", [s1], rsa2048), ("S2", [s2], ec_key), ("S3", [s3, mid], leaf_key)):
        (out / f"{tag}.crt").write_bytes(b"".join(pem_cert(c) for c in chain))  # 서버가 보내는 체인(루트 제외)
        (out / f"{tag}.key").write_bytes(pem_key(key))


# ---------- 로그 ----------
class Log:
    """JSONL 한 줄씩. 여러 스레드에서 안전."""

    def __init__(self, path):
        # "x": 같은 경로로 다시 실행하면 이전 행이 섞여 쿼터·offset 검사가 오염되므로 새 파일만 허용한다
        self.file, self.lock = open(path, "x", encoding="utf-8"), threading.Lock()

    def write(self, **row):
        with self.lock:
            self.file.write(json.dumps(row) + "\n")
            self.file.flush()


# ---------- 서버 ----------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, code, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, PAGE if self.path in ("/", "/login") else b"<html><body>page " + self.path.encode() + b"</body></html>")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        ok = f"u={USER}&p={PASSWORD}" == body
        self.reply(200 if ok else 401, b"welcome" if ok else b"denied", [("Set-Cookie", "sid=1")] if ok else [])


class TlsServer(ThreadingHTTPServer):
    """TCP accept 직후(핸드셰이크 전) 수락 로그를 남기고, 핸드셰이크는 연결 스레드에서 한다(독립 기준 사건 = accept)."""

    def __init__(self, addr, context, accept_log):
        super().__init__(addr, Handler)
        self.context, self.accept_log = context, accept_log

    def get_request(self):
        sock, addr = self.socket.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 헤더·본문 두 레코드가 Nagle·지연 ACK로 늦어지지 않게
        if self.accept_log is not None:
            self.accept_log.write(client_ip=addr[0], client_port=addr[1], server_port=self.server_address[1], t_accept=time.time())
        return sock, addr

    def finish_request(self, request, client_address):
        try:
            request = self.context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            if self.accept_log is not None:  # 실패한 핸드셰이크는 accept 행과 짝을 이루도록 따로 남긴다
                self.accept_log.write(event="hs_fail", client_ip=client_address[0], client_port=client_address[1],
                                      server_port=self.server_address[1], t=time.time())
            return
        try:
            super().finish_request(request, client_address)
        finally:
            request.close()


def serve(certs, bind, accept_log_path=None):
    accept_log = Log(accept_log_path) if accept_log_path else None
    for tag, port in SERVERS.items():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(certs / f"{tag}.crt", certs / f"{tag}.key")
        httpd = TlsServer((bind, port), context, accept_log)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"{tag} {bind}:{port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


# ---------- 러너 ----------
def curl_path():
    """Windows는 PATH보다 System32 curl(Schannel)을 먼저 찾으므로 절대 경로로 고정한다."""
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "curl.exe")
    return system32 if sys.platform == "win32" and os.path.exists(system32) else "curl"


def run_requests(host, port, reqs, rng, passwords, conns):
    """requests 세션 하나 = 에피소드 하나. 쿠키 저장을 끈다(정상에만 생기는 Cookie 헤더 비대칭 방지). 요청 사이 U(0.02,0.2)초.
    훅이 전역을 바꾸므로 러너는 단일 스레드로만 쓴다(병렬화 금지)."""
    import http.cookiejar
    import requests
    import urllib3
    from urllib3.connection import HTTPSConnection

    original, out = HTTPSConnection.connect, []

    def connect(self):
        t0 = time.time()
        original(self)
        local = self.sock.getsockname()
        conns.append(dict(client_ip=local[0], client_port=local[1], t_start=t0))

    HTTPSConnection.connect = connect
    urllib3.disable_warnings()
    try:
        base = f"https://{host}:{port}"
        with requests.Session() as session:
            session.verify = False
            session.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
            for i, (req, pw) in enumerate(zip(reqs, passwords)):
                if i:
                    time.sleep(rng.uniform(0.02, 0.2))
                resp = session.get(base + req["path"], timeout=10) if req["method"] == "GET" else \
                    session.post(base + req["path"], data=dict(u=USER, p=pw), timeout=10)
                out.append((resp.status_code, req["method"] == "POST" and resp.status_code == 200 and resp.content == b"welcome"))
    finally:
        HTTPSConnection.connect = original
    return out, ""


def run_curl(host, port, reqs, rng, passwords, conns):
    """curl 한 번 호출 = 에피소드 하나(--next로 한 연결에 여러 요청, 지연 없이 연속). 로컬 주소·응답 코드·본문 크기는 -w로 받는다.
    --next는 앞 요청의 옵션을 모두 초기화하므로 요청마다 같은 옵션을 다시 준다(연결은 같은 프로세스 안에서 재사용됨)."""
    null = "NUL" if sys.platform == "win32" else "/dev/null"
    cmd = [curl_path()]
    for i, (req, pw) in enumerate(zip(reqs, passwords)):
        if i:
            cmd.append("--next")
        cmd += ["-sSk", "--ssl-no-revoke", "--tlsv1.3", "--connect-timeout", "5", "--max-time", "10", "-o", null,
                "-w", "%{local_ip} %{local_port} %{http_code} %{size_download}" + chr(10)]
        if req["method"] == "POST":
            cmd += ["-d", f"u={USER}&p={pw}"]
        cmd.append(f"https://{host}:{port}{req['path']}")
    t0, out, note = time.time(), [], ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stdout, rc, err = proc.stdout, proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as exc:  # 부분 출력(이미 맺은 연결의 포트)을 보존한다
        raw = exc.stdout or ""
        stdout, rc, err = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw, -1, "timeout"
    lines = [l.split() for l in stdout.strip().splitlines() if l.strip()]
    for (ip, cport, code, size), req in zip([l for l in lines if len(l) == 4], reqs):
        if not any(c["client_port"] == int(cport) for c in conns):
            conns.append(dict(client_ip=ip, client_port=int(cport), t_start=t0))
        out.append((int(code), req["method"] == "POST" and code == "200" and size == "7"))
    note = f"rc={rc} {err.strip()[:80]}" if rc else ""
    return out, note


def episode_problem(reqs, conns, res, note=""):
    """수락 판정: 연결 1개, 응답 수 일치, 상태 코드(GET 200 / POST 200·401), 인증 결과가 배정(expect_ok)과 일치, curl 종료 코드 0. 문제 설명(없으면 "")."""
    if note:
        return note
    if len(conns) != 1 or len(res) != len(reqs):
        return f"conns={len(conns)} responses={len(res)}"
    for r, (code, ok) in zip(reqs, res):
        if code != 200 and not (r["method"] == "POST" and code == 401):
            return f"status={code} {r['method']}"
        if ok != r["expect_ok"]:
            return f"auth={ok} expected={r['expect_ok']}"
    return ""


def do_episode(impl, host, port, reqs, rng, conns):
    passwords = [password_for(r, rng) for r in reqs]
    res, note = (run_requests if impl == "requests" else run_curl)(host, port, reqs, rng, passwords, conns)
    return res, passwords, note


MAX_ATTEMPTS, MAX_CONSECUTIVE_FAILS = 10, 20


def run_phase(args, server, phase, plan, rng, log, attempts):
    """한 서버·한 단계: plan(미리 생성한 라운드별 에피소드 목록)을 수행한다. 문제가 있는 시도는 제외하고 다시 수행하되(최대 MAX_ATTEMPTS, 실패마다 대기 증가),
    쿼터를 못 채우거나 연속 실패가 쌓이면 중단한다(수량 기준을 낮추지 않음). 모든 시도를 attempts에 남긴다(실패 시도의 클라이언트 포트 포함)."""
    port, ep_no, last_end, consecutive = SERVERS[server], 0, 0.0, 0
    for rnd, episodes in enumerate(plan, 1):
        for cls, impl, first, m, subset, failonly in episodes:
            ep_type, reqs = browse_plan() if subset == "browse" else episode_plan(cls, first, m, failonly)
            ep_no += 1
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t_ep, conns = time.time(), []  # conns는 예외가 나도 그때까지 맺은 연결(포트)을 남긴다
                try:
                    res, passwords, note = do_episode(impl, args.host, port, reqs, rng, conns)
                    reason = episode_problem(reqs, conns, res, note)
                except Exception as exc:  # 연결 실패·응답 시간 초과 등: 제외하고 재수행
                    res, passwords, reason = [], [], f"error:{type(exc).__name__}"
                t_end = time.time()
                key = dict(phase=phase, round=rnd, episode=ep_no, attempt=attempt, cls=cls, impl=impl, subset=subset, ep_type=ep_type, m=len(reqs),
                           first=reqs[0]["method"], server_name=server)
                attempts.write(**key, accepted=not reason, reason=reason, t_begin=t_ep, t_end=t_end, conns=[(c["client_ip"], c["client_port"]) for c in conns])
                time.sleep(rng.uniform(0.1, 0.5) * (1 if not reason else attempt))  # 에피소드 사이 간격(모든 칸 동일), 실패하면 점점 길게
                if not reason:
                    consecutive = 0
                    log.write(**key, client_ip=conns[0]["client_ip"], client_port=conns[0]["client_port"], server_ip=args.host, server_port=port,
                              t_start=conns[0]["t_start"], t_end=t_end, conns_in_episode=1, episode_requests=len(reqs),
                              kinds="".join(r["method"][0] for r in reqs), pw_lens=[None if p is None else len(p) for p in passwords],
                              statuses=[r[0] for r in res], auth_ok=[r[1] for r in res])
                    last_end = max(last_end, t_end)
                    break
                consecutive += 1
                if consecutive > MAX_CONSECUTIVE_FAILS:
                    raise RuntimeError(f"{phase} {server} round {rnd}: 연속 {consecutive}회 실패({reason}). 서버·네트워크를 확인하세요(쿼터를 낮추지 않고 중단)")
            else:
                raise RuntimeError(f"{phase} {server} round {rnd} episode {ep_no}: {MAX_ATTEMPTS}회 시도에도 실패({reason}). 쿼터를 못 채워 중단")
    return last_end


def git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True, encoding="utf-8", cwd=Path(__file__).resolve().parents[2]).strip()
    except Exception:
        return None


def run_metadata(args):
    ver = lambda cmd: (subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines() or [""])[0]
    try:
        import requests
        import urllib3
        requests_v = f"{requests.__version__} urllib3 {urllib3.__version__}"
    except Exception:
        requests_v = None
    dirty = git("status", "--porcelain", "--", "src", "tests")  # git이 없으면 None(깨끗한지 알 수 없음). false로 기록하지 않는다
    src = Path(__file__).resolve().parent
    sha = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in ("collect.py", "pcap_flows.py") if (src / n).exists()}
    return dict(git_commit=git("rev-parse", "HEAD"), git_dirty_src_tests=None if dirty is None else bool(dirty), files_sha256=sha, seed=args.seed,
                host=args.host, python=platform.python_version(), openssl=ssl.OPENSSL_VERSION, requests=requests_v, curl_path=curl_path(),
                curl=ver([curl_path(), "-V"]), platform=platform.platform(), train_rounds=args.train_rounds, test_rounds=args.test_rounds,
                wait_after=args.wait_after, pad=args.pad, started=dt.datetime.now().isoformat(timespec="seconds"))


def run(args):
    import random
    log, attempts = Log(args.log), Log(args.attempts)
    Path(args.log).with_suffix(".meta.json").write_text(json.dumps(run_metadata(args), indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    sched, rng = random.Random(args.seed), random.Random(args.seed + 1)  # 일정(라운드 순서)과 암호·간격을 분리: 재시도가 있어도 일정은 같다
    plans = {(phase, server): [round_episodes(sched, rnd) for rnd in range(rounds)]
             for phase, rounds in (("train", args.train_rounds), ("test", args.test_rounds)) for server in SERVERS}
    last_end = {}
    for phase in ("train", "test"):
        for server in SERVERS:
            if phase == "test":  # 시험 직전: 그 서버의 마지막 학습·지원 에피소드 종료 + wait_after초까지 대기
                wait = last_end[server] + args.wait_after - time.time()
                print(f"{server} test: 학습 종료 후 {args.wait_after}초까지 {max(wait, 0):.0f}초 대기", flush=True)
                if wait > 0:
                    time.sleep(wait)
            end = run_phase(args, server, phase, plans[(phase, server)], rng, log, attempts)
            if phase == "train":
                last_end[server] = end
            print(f"{phase} {server}: 완료", flush=True)
            time.sleep(args.pad)
    print(f"run done -> {args.log}", flush=True)


# ---------- 캡처 ----------
def capture(args):
    flt = " or ".join(f"tcp port {port}" for port in SERVERS.values())
    cmd = [DUMPCAP, "-i", args.iface, "-f", flt, "-s", "128", "-F", "pcap", "-w", str(args.out)]
    if args.minutes:
        cmd += ["-a", f"duration:{int(args.minutes * 60)}"]
    subprocess.run(cmd, check=True)


# ---------- 매칭 ----------
MSS = 1460  # 이더넷 MSS(Windows는 TCP timestamps를 쓰지 않아 1460). 이보다 큰 payload가 보이면 LSO·RSC로 합쳐진 것(1448은 timestamps 사용 시 값이라 틀림)
MIN_REF = 30  # 시계 offset 기준 연결 수 최소값(미달이면 파일럿 불합격, offset=0으로 대체하지 않음)
MAD_MAX_S, LOST_MAX, MIN_GAP_S = 0.2, 0.02, 600.0  # 파일럿 승인 기준(문서 3절·2절)
EXTRA_MAX, ROUND_RESID_MAX_S = 0.02, 0.2  # 여분 플로우 비율 상한, 라운드별 시계 잔차 중앙값 상한(초)


def expected_groups(train_rounds, test_rounds):
    """실행 설정이 만드는 모든 (서버, 단계) 그룹과 (서버, 단계, 라운드, 구현, 클래스) 칸."""
    phases = (("train", train_rounds), ("test", test_rounds))
    return ([(ph, sv) for ph, _ in phases for sv in SERVERS],
            [(sv, ph, rd, im, cl) for ph, n in phases for sv in SERVERS for rd in range(1, n + 1) for im in ("requests", "curl") for cl in ("normal", "attack")])


def key4(row):
    return row["client_ip"], row["client_port"], row["server_ip"], row["server_port"]


def unique_pairs(logs, server_log):
    """(client_ip, client_port, server_port)가 러너 로그와 서버 수락 로그 양쪽에서 유일한 연결의 (러너 행, 서버 행) 목록."""
    def group(rows):
        out = {}
        for r in rows:
            out.setdefault((r["client_ip"], r["client_port"], r["server_port"]), []).append(r)
        return out
    a, b = group(logs), group(server_log)
    return [(a[k][0], b[k][0]) for k in a if k in b and len(a[k]) == 1 and len(b[k]) == 1]


def estimate_offset(logs, server_log, min_ref=MIN_REF):
    """runner–server 정렬 offset = 중앙값(서버 accept 시각 − runner t_start). 시간 허용오차 필터 없이 양쪽에서 유일한 연결만 쓴다.
    서로 다른 사건의 지연을 포함하므로 순수 시계 오차가 아니다. 반환 (offset, 잔차 리스트, 기준 연결 수). 기준 연결이 부족하면 예외(0으로 대체하지 않음)."""
    diffs = [s["t_accept"] - r["t_start"] for r, s in unique_pairs(logs, server_log)]
    if len(diffs) < min_ref:
        raise ValueError(f"파일럿 불합격: runner–server 기준 연결 {len(diffs)}개 < {min_ref} (offset=0으로 대체하지 않음)")
    offset = statistics.median(diffs)
    return offset, [d - offset for d in diffs], len(diffs)


def group_offsets(logs, server_log, min_ref=MIN_REF, expected=None):
    """(단계, 서버)별 offset과 보정 후 잔차(전체·구현별·라운드별 중앙값). expected(예상 그룹 목록)의 모든 그룹에 기준 연결이 min_ref개 이상 있어야 하며,
    하나라도 없거나 모자라면 예외(전체 offset으로 대체하지 않음)."""
    groups = {}
    for r, s in unique_pairs(logs, server_log):
        groups.setdefault((r["phase"], r["server_name"]), []).append((r, s["t_accept"] - r["t_start"]))
    for key in (expected if expected is not None else groups):
        if len(groups.get(key, [])) < min_ref:
            raise ValueError(f"파일럿 불합격: {key} 기준 연결 {len(groups.get(key, []))}개 < {min_ref} (전체 offset으로 대체하지 않음)")
    out = {}
    for key, items in groups.items():
        off = statistics.median(d for _, d in items)
        res = [(r, d - off) for r, d in items]
        med = lambda xs: round(statistics.median(xs), 4)
        out[key] = dict(offset=off, n=len(items), mad=med(abs(x) for _, x in res),
                        resid_by_impl={i: med(x for r, x in res if r["impl"] == i) for i in {r["impl"] for r, _ in res}},
                        resid_by_round={int(k): med(x for r, x in res if r["round"] == k) for k in sorted({r["round"] for r, _ in res})})
    return out


def match(flows, logs, offset, tolerance=1.0):
    """flows: pcap_flows.Flow 목록(서버 시계), logs: 러너 로그, offset: runner→server 시계 보정(숫자 또는 행→숫자 함수).
    같은 4-튜플이고 |flow.start_ts − (t_start+offset)| ≤ tolerance인 후보가 정확히 하나일 때만 매칭한다. 후보가 둘 이상이면 **모호**로 분리하고(임의 최근접 선택 금지),
    로그 한 줄을 둘 이상의 플로우가 주장해도 모호로 둔다. 반환: (matched [(flow, log)], unmatched_flows, unmatched_logs, ambiguous [(flow, [logs])])."""
    off = offset if callable(offset) else (lambda r: offset)
    by_key = {}
    for row in logs:
        by_key.setdefault(key4(row), []).append(row)
    claims, cand_of = {}, {}
    for idx, flow in enumerate(flows):
        cands = [r for r in by_key.get((flow.client_ip, flow.client_port, flow.server_ip, flow.server_port), [])
                 if abs(flow.start_ts - (r["t_start"] + off(r))) <= tolerance]
        cand_of[idx] = cands
        for r in cands:
            claims.setdefault(id(r), []).append(idx)
    matched, unmatched, ambiguous, used = [], [], [], set()
    for idx, flow in enumerate(flows):
        cands = cand_of[idx]
        if not cands:
            unmatched.append(flow)
        elif len(cands) > 1 or len(claims[id(cands[0])]) > 1:
            ambiguous.append((flow, cands))
            used.update(id(r) for r in cands)
        else:
            matched.append((flow, cands[0]))
            used.add(id(cands[0]))
    return matched, unmatched, [r for r in logs if id(r) not in used], ambiguous


def first_request_index(seq):
    """첫 HTTP 요청의 패킷 위치 **후보** 방법: ClientHello(seq_0) 뒤 첫 양수 칸 = 클라이언트 Finished, 그다음 양수 칸 = 첫 요청 후보.
    구현·서버에 따라 티켓/요청 순서가 바뀌므로 확정이 아니다. 찾지 못하면 None(미확인). 서버가 요청 시각을 기록하지 않아 시각 대응은 하지 않는다."""
    pos = [i for i, v in enumerate(list(seq)) if i >= 1 and v > 0]
    return pos[1] if len(pos) >= 2 else None


def check_quota(rows, expected_cells=None):
    """주(main) 연결 쿼터·배정표 일치 검사. 라운드·서버·단계·칸마다 정확히 배정표의 10연결((first, m, 요청 종류, 암호 길이) 일치), 정상 7/2/1,
    인증 결과가 배정(expect_ok)과 일치, 같은 (first,m)에서 클래스 간 요청 종류·암호 길이 동일. 위반 목록 반환."""
    bad, cells = [], {}
    for r in rows:
        if r.get("subset") == "main":
            cells.setdefault((r["server_name"], r["phase"], r["round"], r["impl"], r["cls"]), []).append(r)
    ref = {}
    for cls in ("normal", "attack"):
        plans = [(f, m, episode_plan(cls, f, m)) for f, m in CELL_SPECS]
        ref[cls] = sorted((f, m, "".join(q["method"][0] for q in reqs), tuple(q["pw_len"] for q in reqs)) for f, m, (_, reqs) in plans)
    # 인증 결과 기준은 에피소드 유형(success/failonly/none/fail)별로 배정표에서 다시 만든다
    auth = {(cls, f, m, t): [q["expect_ok"] for q in episode_plan(cls, f, m, t == "failonly")[1]]
            for cls in ("normal", "attack") for f, m in CELL_SPECS for t in ("success", "failonly", "none", "fail")}
    for k in (expected_cells or []):
        if k not in cells:
            bad.append(f"{k}: 칸이 비어 있음(연결 0)")
    for k, rs in cells.items():
        got = sorted((r["first"], r["m"], r["kinds"], tuple(r["pw_lens"])) for r in rs)
        if got != ref[k[4]]:
            bad.append(f"{k}: (first,m,종류,암호 길이) 배정이 기준표와 다름(연결 {len(rs)})")
        for r in rs:
            if list(r["auth_ok"]) != auth.get((k[4], r["first"], r["m"], r["ep_type"])):
                bad.append(f"{k}: 인증 결과가 배정과 다름 episode={r['episode']}")
        if k[4] == "normal":
            n = {t: sum(r["ep_type"] == t for r in rs) for t in ("success", "failonly", "none")}
            if n != {"success": 7, "failonly": 2, "none": 1}:
                bad.append(f"{k}: 정상 구성 {n} != 7/2/1")
    # 실패 전용이 특정 칸에 몰리지 않아야 한다: (서버, 단계, 구현)마다 칸별 횟수 차이 ≤1
    fo = {}
    for (sv, ph, _, impl, cls), rs in cells.items():
        if cls == "normal":
            c = fo.setdefault((sv, ph, impl), {cell: 0 for cell in LOGIN_CELLS})
            for r in rs:
                if r["ep_type"] == "failonly":
                    c[(r["first"], r["m"])] = c.get((r["first"], r["m"]), 0) + 1
    for k, c in fo.items():
        if max(c.values()) - min(c.values()) > 1:
            bad.append(f"{k}: 실패 전용이 칸에 치우침 {sorted(c.items())}")
    return bad


def breakdown(logs, matched=(), ambiguous=(), offset=0.0):
    """(서버, 단계, cls, impl, subset)별 분모: 연결 수·에피소드 수·요청 수·인증 성공/실패·매칭·모호·미매칭·시각 오차 최댓값(보정 후, 초)·손실 비율."""
    off = offset if callable(offset) else (lambda r: offset)
    hit = {id(r): flow for flow, r in matched}
    amb = {id(r) for _, rs in ambiguous for r in rs}
    out = {}
    for r in logs:
        d = out.setdefault(f"{r.get('server_name')}/{r.get('phase')}/{r['cls']}/{r['impl']}/{r.get('subset', 'main')}", dict(conns=0, episodes=set(), requests=0, auth_ok=0, auth_fail=0,
                                                                                     matched=0, ambiguous=0, unmatched=0, max_resid_s=0.0))
        d["conns"] += 1
        d["episodes"].add((r.get("phase"), r.get("server_name"), r.get("episode")))  # 에피소드 번호는 단계·서버마다 다시 시작한다
        d["requests"] += r.get("episode_requests", 1)
        posts = [k == "P" for k in r.get("kinds", "")]
        d["auth_ok"] += sum(bool(ok) for ok, p in zip(r.get("auth_ok", []), posts) if p)
        d["auth_fail"] += sum(not ok for ok, p in zip(r.get("auth_ok", []), posts) if p)
        if id(r) in hit:
            d["matched"] += 1
            d["max_resid_s"] = max(d["max_resid_s"], abs(hit[id(r)].start_ts - (r["t_start"] + off(r))))
        elif id(r) in amb:
            d["ambiguous"] += 1
        else:
            d["unmatched"] += 1
    for d in out.values():
        d["episodes"] = len(d["episodes"])
        d["max_resid_s"] = round(d["max_resid_s"], 3)
        d["requests_per_conn"] = round(d["requests"] / d["conns"], 2)
        d["lost_frac"] = round((d["ambiguous"] + d["unmatched"]) / d["conns"], 4)  # (서버, 단계, 구현, 클래스)별 (모호+미매칭) 비율: 파일럿 합격 ≤0.02 (합쳐서 희석하지 않음)
    return out


def test_gaps(logs):
    """서버별 (시험 첫 연결 시작 − 학습 마지막 에피소드 종료) 초. 문서 2절: ≥600."""
    out = {}
    for server in SERVERS:
        tr = [r["t_end"] for r in logs if r["server_name"] == server and r["phase"] == "train"]
        te = [r["t_start"] for r in logs if r["server_name"] == server and r["phase"] == "test"]
        if tr and te:
            out[server] = round(min(te) - max(tr), 1)
    return out


def read_jsonl(paths):
    return [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_clean(skipped, over_mss, hs_fail, matched):
    """건너뛴 패킷 없음·MSS 초과 0·핸드셰이크 실패 0·매칭된 플로우 중 중간 시작 없음(skipped는 딕셔너리라 `== 0`이 아니라 비어 있는지로 본다)."""
    return not skipped and over_mss == 0 and hs_fail == 0 and not any(flow.midstream for flow, _ in matched)


def match_main(args):
    from src.track_b.pcap_flows import K_MAX, assemble_flows
    flows, skipped, tcp_packets = assemble_flows(args.pcap, ports=tuple(SERVERS.values()))
    logs, raw_server = read_jsonl(args.logs), read_jsonl([args.server_log])
    server_log = [r for r in raw_server if "t_accept" in r]
    hs_fail = sum(r.get("event") == "hs_fail" for r in raw_server)
    meta_path = Path(args.logs[0]).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    train_r, test_r = args.train_rounds or meta.get("train_rounds"), args.test_rounds or meta.get("test_rounds")
    if train_r is None or test_r is None:
        raise ValueError("예상 라운드 수를 알 수 없습니다: 러너 .meta.json이 없으면 --train-rounds/--test-rounds를 주세요")
    exp_groups, exp_cells = expected_groups(train_r, test_r)
    offset, resid, n_ref = estimate_offset(logs, server_log, args.min_ref)
    goff = group_offsets(logs, server_log, args.min_ref, exp_groups)  # 예상 그룹마다 기준 연결이 있어야 한다(없으면 불합격, 전체 offset 대체 없음)
    off_of = lambda r: goff[(r["phase"], r["server_name"])]["offset"]
    matched, unmatched_flows, unmatched_logs, ambiguous = match(flows, logs, off_of, args.tolerance)
    cnt = {}
    for r in server_log:
        k = (r["client_ip"], r["client_port"], r["server_port"])
        cnt[k] = cnt.get(k, 0) + 1
    accept = {(r["client_ip"], r["client_port"], r["server_port"]): r["t_accept"] for r in server_log}
    lag = [flow.start_ts - accept[k] for flow, _ in matched for k in [(flow.client_ip, flow.client_port, flow.server_port)] if cnt.get(k) == 1]
    over_mss = sum(any(abs(v) > MSS for v in flow.seq) for flow in flows)
    ks = (5, 10, 15)
    fields = ["server", "cls", "impl", "subset", "ep_type", "phase", "round", "episode", "client_ip", "client_port", "server_ip", "server_port", "start_ts",
              "payload_packets", "midstream", "first_req_idx"] + [f"t_k{k}" for k in ks] + [f"seq_{i}" for i in range(K_MAX)]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for flow, row in matched:
            seq = list(flow.seq) + [0] * (K_MAX - len(flow.seq))
            idx = first_request_index(flow.seq)
            tk = {f"t_k{k}": (round(flow.times[k - 1], 6) if len(flow.times) >= k else "") for k in ks}  # 연결 시작→k번째 payload 관측 대기시간(모델 처리시간 아님)
            writer.writerow(dict(server=row["server_name"], cls=row["cls"], impl=row["impl"], subset=row["subset"], ep_type=row["ep_type"], phase=row["phase"],
                                 round=row["round"], episode=row["episode"], client_ip=flow.client_ip, client_port=flow.client_port, server_ip=flow.server_ip,
                                 server_port=flow.server_port, start_ts=flow.start_ts, payload_packets=flow.payload_packets, midstream=int(flow.midstream),
                                 first_req_idx="" if idx is None else idx, **tk, **{f"seq_{i}": seq[i] for i in range(K_MAX)}))
    p99 = lambda xs: sorted(abs(x) for x in xs)[min(len(xs) - 1, int(0.99 * len(xs)))] if xs else None
    by_label = breakdown(logs, matched, ambiguous, off_of)
    matched_rows = [r for _, r in matched]
    gaps = test_gaps(logs)
    main_cells = {k: v for k, v in by_label.items() if k.endswith("/main")}
    n_expected_labels = len(SERVERS) * 2 * 2 * 2  # 서버 × 단계 × 구현 × 클래스
    pilot = dict(mad_ok=all(g["mad"] <= MAD_MAX_S for g in goff.values()),
                 lost_ok=len(main_cells) == n_expected_labels and all(v["lost_frac"] <= LOST_MAX for v in main_cells.values()),
                 quota_ok=not check_quota(logs, exp_cells),
                 gap_ok=set(gaps) == set(SERVERS) and all(v >= MIN_GAP_S for v in gaps.values()),
                 extra_ok=len(unmatched_flows) <= EXTRA_MAX * max(len(flows), 1),  # 로그에 없는 여분 연결(다른 접속·재시도 누락 등)
                 clean_ok=is_clean(skipped, over_mss, hs_fail, matched),
                 round_clock_ok=all(abs(x) <= ROUND_RESID_MAX_S for g in goff.values() for x in g["resid_by_round"].values()))  # 실행 중 시계가 튄 라운드 검출
    report = dict(flows=len(flows), tcp_packets=tcp_packets, pcap_skipped=skipped, flows_with_segment_over_mss=over_mss, runner_rows=len(logs),
                  server_log_rows=len(server_log), server_handshake_failures=hs_fail, matched=len(matched), unmatched_flows=len(unmatched_flows),
                  unmatched_logs=len(unmatched_logs), ambiguous=len(ambiguous), by_label=by_label,
                  clock=dict(offset_runner_to_server_s=round(offset, 4), ref_connections=n_ref, resid_mad_s=round(statistics.median(abs(r) for r in resid), 4),
                             resid_p99_s=round(p99(resid), 4), by_group={f"{k[0]}/{k[1]}": {**v, "offset": round(v["offset"], 4)} for k, v in goff.items()},
                             note="필터 전 유일 대응 연결 기준. 서로 다른 사건(accept vs connect 직전)의 지연을 포함하므로 순수 시계오차가 아님"),
                  pcap_minus_accept_s=dict(median=round(statistics.median(lag), 4) if lag else None, p99=round(p99(lag), 4) if lag else None),
                  test_gap_s=gaps, quota_violations=check_quota(logs, exp_cells), quota_violations_matched_rows=check_quota(matched_rows, exp_cells)[:20], pilot_checks=pilot,
                  tolerance_s=args.tolerance, created=dt.datetime.now().isoformat(timespec="seconds"))
    Path(args.out).with_suffix(".match.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(json.dumps(dict(pilot_checks=pilot, matched=len(matched), ambiguous=len(ambiguous), unmatched_flows=len(unmatched_flows), skipped=skipped,
                          over_mss=over_mss, test_gap_s=gaps), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("certs")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("serve")
    p.add_argument("--certs", type=Path, required=True)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--accept-log", type=Path, help="연결 수락 로그(JSONL). 매칭의 독립 기준")
    p = sub.add_parser("run")
    p.add_argument("--host", required=True)
    p.add_argument("--log", type=Path, required=True, help="러너 로그(accepted 에피소드, 연결 1개당 한 줄)")
    p.add_argument("--attempts", type=Path, required=True, help="전체 시도 기록(제외·재수행 포함)")
    p.add_argument("--train-rounds", type=int, default=30)
    p.add_argument("--test-rounds", type=int, default=10)
    p.add_argument("--wait-after", type=float, default=660, help="시험 직전 대기: 그 서버 마지막 학습 에피소드 종료 + 이 초")
    p.add_argument("--pad", type=float, default=60, help="단계 사이 대기(초)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("capture")
    p.add_argument("--iface", required=True, help="dumpcap -D 의 번호 또는 이름")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--minutes", type=float, default=0)
    p = sub.add_parser("match")
    p.add_argument("--pcap", type=Path, required=True)
    p.add_argument("--logs", type=Path, nargs="+", required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--min-ref", type=int, default=MIN_REF)
    p.add_argument("--train-rounds", type=int, help="예상 학습 라운드 수(기본: 러너 .meta.json)")
    p.add_argument("--test-rounds", type=int, help="예상 시험 라운드 수(기본: 러너 .meta.json)")
    args = parser.parse_args()
    {"certs": lambda: make_certs(args.out), "serve": lambda: serve(args.certs, args.bind, args.accept_log), "run": lambda: run(args),
     "capture": lambda: capture(args), "match": lambda: match_main(args)}[args.cmd]()


if __name__ == "__main__":
    main()
