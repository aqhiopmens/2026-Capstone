"""안건 12 통제 수집: 서버 3개(S1~S3) 구동, 클라이언트 러너, dumpcap 캡처, pcap 플로우-러너 로그 매칭.

라벨은 러너 로그의 (client_ip, client_port, server_ip, server_port) + 시각에서만 가져온다.
IP·서버로 라벨을 정하지 않는다. 자기 서버·자기 LAN에만 쓴다(외부 대상 없음).

  certs   인증서 생성 (S1 RSA-2048 단일, S2 ECDSA P-256, S3 RSA-4096 2단 체인)
  serve   서버 3개 구동 (기본 127.0.0.1, LAN 수집 때만 --bind 0.0.0.0)
  run     클라이언트 러너 (정상·공격, requests·curl 번갈아). 연결마다 로그 한 줄
  capture dumpcap 래퍼 (classic pcap)
  match   pcap 플로우 + 러너 로그 -> 라벨 붙은 flows.csv (시계 어긋남 보정)
"""
import argparse
import csv
import datetime as dt
import json
import ssl
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVERS = {"S1": 8443, "S2": 8444, "S3": 8445}
USER, PASSWORD = "admin", "correct-horse-9"
DUMPCAP = r"C:\Program Files\Wireshark\dumpcap.exe"
PAGE = b"<html><body><h1>Portal</h1><form method=post action=/login>user <input name=u> pass <input name=p type=password></form></body></html>"


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
        self.reply(200 if ok else 401, b"welcome" if ok else b"denied",
                   [("Set-Cookie", "sid=1")] if ok else [])


def serve(certs, bind):
    servers = []
    for tag, port in SERVERS.items():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(certs / f"{tag}.crt", certs / f"{tag}.key")
        httpd = ThreadingHTTPServer((bind, port), Handler)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        print(f"{tag} {bind}:{port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


# ---------- 러너 ----------
class Log:
    """연결마다 한 줄: 4-튜플·시작 시각(연결 직전)·구현·클래스·서버. 여러 스레드에서 안전."""

    def __init__(self, path, server):
        self.file, self.server, self.lock = open(path, "a", encoding="utf-8"), server, threading.Lock()

    def write(self, **row):
        with self.lock:
            self.file.write(json.dumps(dict(server=self.server, **row)) + "\n")
            self.file.flush()


def _ok(code):
    return 200 <= int(code) < 300


def requests_visit(log, host, port, cls, passwords, ep):
    """requests 세션 하나 = 에피소드 하나. 세션을 끊고 나서야 다음 에피소드가 시작되므로 한 연결에 정상·공격이 섞이지 않는다.
    연결 시점은 urllib3 connect 후킹으로 기록하고, 요청 수·성공/실패는 에피소드가 끝난 뒤 연결 행에 같이 쓴다."""
    import requests
    import urllib3
    from urllib3.connection import HTTPSConnection

    original, conns, codes = HTTPSConnection.connect, [], []

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
            for path in (["/", "/login"] if cls == "normal" else ["/login"]):
                codes.append(session.get(base + path, timeout=10).status_code)
            for password in passwords:
                codes.append(session.post(base + "/login", data=dict(u=USER, p=password), timeout=10).status_code)
                time.sleep(0.05 if cls == "normal" else 0)
    finally:
        HTTPSConnection.connect = original
    for c in conns:  # 연결 재사용 여부는 episode_requests / conns_in_episode 로 사후 확인
        log.write(**c, server_ip=host, server_port=port, impl="requests", cls=cls, episode=ep,
                  conns_in_episode=len(conns), episode_requests=len(codes),
                  episode_ok=sum(map(_ok, codes)), episode_fail=len(codes) - sum(map(_ok, codes)))


def curl_call(log, host, port, cls, method, path, ep, n_calls, data=None):
    """curl 호출 하나 = 연결 하나 = 요청 하나. 로컬 주소·응답 코드는 curl -w 로 받는다."""
    cmd = ["curl", "-sk", "--ssl-no-revoke", "--tlsv1.3", "-o", "NUL" if sys.platform == "win32" else "/dev/null",
           "-w", "%{local_ip} %{local_port} %{http_code}", f"https://{host}:{port}{path}"]
    if method == "POST":
        cmd += ["-d", data]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.split()
    if len(out) == 3:
        ok = _ok(out[2]) if out[2].isdigit() and int(out[2]) else False
        log.write(client_ip=out[0], client_port=int(out[1]), server_ip=host, server_port=port, t_start=t0,
                  impl="curl", cls=cls, episode=ep, conns_in_episode=n_calls, episode_requests=n_calls,
                  http_code=out[2], ok=ok)


def curl_visit(log, host, port, cls, passwords, ep):
    calls = ([("GET", "/", None)] if cls == "normal" else []) + [("POST", "/login", f"u={USER}&p={p}") for p in passwords]
    for method, path, data in calls:
        curl_call(log, host, port, cls, method, path, ep, len(calls), data)
        time.sleep(0.05 if cls == "normal" and method == "POST" else 0)


def passwords_for(cls, rng, attack_burst):
    """정상: 올바른 로그인 앞에 오타 0~3회 + 몇 번 탐색. 공격: 목록 순서대로 attack_burst개(정답 없음)."""
    if cls == "normal":
        return [f"typo{rng.randint(0, 99)}" for _ in range(rng.choice([0, 0, 1, 2, 3]))] + [PASSWORD]
    return [f"pw{rng.randint(0, 99999):05d}" for _ in range(attack_burst)]


def run(args):
    import random
    rng = random.Random(args.seed)
    log = Log(args.log, args.server)
    port = SERVERS[args.server]
    deadline = time.time() + args.minutes * 60
    combos = [(c, i) for c in ("normal", "attack") for i in ("requests", "curl")]
    n = ep = 0
    while time.time() < deadline:
        cls, impl = combos[n % 4]  # 같은 기기·IP에서 번갈아
        n += 1
        visit = requests_visit if impl == "requests" else curl_visit
        if cls == "normal":  # 정상 방문 여러 번(클래스 간 연결 수 균형)
            for _ in range(args.normal_visits):
                ep += 1
                visit(log, args.host, port, cls, passwords_for(cls, rng, 0), ep)
        else:  # requests는 세션당 로그인 N회로 끊어 연결(=플로우) 수를 늘린다. curl은 원래 호출마다 새 연결
            passwords = passwords_for(cls, rng, args.attack_burst)
            step = args.logins_per_session if impl == "requests" else len(passwords)
            for i in range(0, len(passwords), step):
                ep += 1
                visit(log, args.host, port, cls, passwords[i:i + step], ep)
        time.sleep(rng.uniform(0.1, 1.0) if cls == "normal" else 0.1)
    print(f"{args.server}: {n} rounds -> {args.log}", flush=True)


# ---------- 캡처 ----------
def capture(args):
    flt = " or ".join(f"tcp port {port}" for port in SERVERS.values())
    cmd = [DUMPCAP, "-i", args.iface, "-f", flt, "-s", "128", "-F", "pcap", "-w", str(args.out)]
    if args.minutes:
        cmd += ["-a", f"duration:{int(args.minutes * 60)}"]
    subprocess.run(cmd, check=True)


# ---------- 매칭 ----------
def match(flows, logs, tolerance=1.0):
    """flows: pcap_flows.Flow 목록, logs: 러너 로그 dict 목록.

    1) 4-튜플이 같은 후보를 찾고 2) 두 기기 시계 어긋남(offset=중앙값(flow.start_ts - t_start))을 추정한 뒤
    3) 보정 후 시각 차이가 tolerance 초 이내인 가장 가까운 로그 하나를 쓴다. 로그는 한 번만 쓴다.
    반환: (matched [(flow, log)], unmatched_flows, unmatched_logs, offset)
    """
    by_key = {}
    for row in logs:
        by_key.setdefault((row["client_ip"], row["client_port"], row["server_ip"], row["server_port"]), []).append(row)
    pairs = [(flow, by_key[(flow.client_ip, flow.client_port, flow.server_ip, flow.server_port)])
             for flow in flows if (flow.client_ip, flow.client_port, flow.server_ip, flow.server_port) in by_key]
    unique = [flow.start_ts - rows[0]["t_start"] for flow, rows in pairs if len(rows) == 1]
    offset = statistics.median(unique) if unique else 0.0
    used, matched, unmatched = set(), [], []
    for flow in flows:
        rows = by_key.get((flow.client_ip, flow.client_port, flow.server_ip, flow.server_port), [])
        best = min((r for r in rows if id(r) not in used),
                   key=lambda r: abs(flow.start_ts - offset - r["t_start"]), default=None)
        if best is not None and abs(flow.start_ts - offset - best["t_start"]) <= tolerance:
            used.add(id(best))
            matched.append((flow, best))
        else:
            unmatched.append(flow)
    return matched, unmatched, [r for r in logs if id(r) not in used], offset


def breakdown(logs, matched=(), offset=0.0):
    """(cls, impl)별 분모: 로그 연결 수·에피소드 수·요청 수·성공/실패·매칭 수·미매칭 수·시각 오차 최댓값(보정 후, 초).
    요청 수는 에피소드 값을 그 에피소드 연결 수로 나눠 연결마다 더한 것이다(재연결이 있어도 에피소드 합이 보존된다)."""
    hit = {id(r): flow for flow, r in matched}
    out = {}
    for r in logs:
        d = out.setdefault(f"{r['cls']}/{r['impl']}", dict(conns=0, episodes=set(), requests=0.0, ok=0.0, fail=0.0,
                                                           matched=0, unmatched=0, max_resid_s=0.0))
        share = 1 / r.get("conns_in_episode", 1)
        d["conns"] += 1
        d["episodes"].add(r.get("episode"))
        d["requests"] += r.get("episode_requests", 1) * share
        if "episode_ok" in r:
            d["ok"] += r["episode_ok"] * share
            d["fail"] += r["episode_fail"] * share
        else:
            d["ok"] += bool(r.get("ok"))
            d["fail"] += not r.get("ok")
        if id(r) in hit:
            d["matched"] += 1
            d["max_resid_s"] = max(d["max_resid_s"], abs(hit[id(r)].start_ts - offset - r["t_start"]))
        else:
            d["unmatched"] += 1
    for d in out.values():
        d["episodes"] = len(d["episodes"])
        d.update({k: round(d[k], 3) for k in ("requests", "ok", "fail", "max_resid_s")})
        d["requests_per_conn"] = round(d["requests"] / d["conns"], 2)
    return out


def match_main(args):
    from src.track_b.pcap_flows import K_MAX, assemble_flows
    flows, _, _ = assemble_flows(args.pcap, ports=tuple(SERVERS.values()))
    logs = [json.loads(line) for path in args.logs for line in Path(path).read_text(encoding="utf-8").splitlines()]
    matched, unmatched_flows, unmatched_logs, offset = match(flows, logs, args.tolerance)
    fields = ["server", "cls", "impl", "client_ip", "client_port", "server_ip", "server_port", "start_ts",
              "payload_packets", "midstream"] + [f"seq_{i}" for i in range(K_MAX)]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for flow, row in matched:
            seq = list(flow.seq) + [0] * (K_MAX - len(flow.seq))
            writer.writerow(dict(server=row["server"], cls=row["cls"], impl=row["impl"], client_ip=flow.client_ip,
                                 client_port=flow.client_port, server_ip=flow.server_ip, server_port=flow.server_port,
                                 start_ts=flow.start_ts, payload_packets=flow.payload_packets,
                                 midstream=int(flow.midstream), **{f"seq_{i}": seq[i] for i in range(K_MAX)}))
    report = dict(flows=len(flows), logs=len(logs), matched=len(matched), unmatched_flows=len(unmatched_flows),
                  unmatched_logs=len(unmatched_logs), by_label=breakdown(logs, matched, offset), clock_offset_s=round(offset, 4), tolerance_s=args.tolerance,
                  created=dt.datetime.now().isoformat(timespec="seconds"))
    Path(args.out).with_suffix(".match.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("certs")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("serve")
    p.add_argument("--certs", type=Path, required=True)
    p.add_argument("--bind", default="127.0.0.1")
    p = sub.add_parser("run")
    p.add_argument("--host", required=True)
    p.add_argument("--server", choices=SERVERS, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--attack-burst", type=int, default=25)
    p.add_argument("--normal-visits", type=int, default=8)
    p.add_argument("--logins-per-session", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("capture")
    p.add_argument("--iface", required=True, help="dumpcap -D 의 번호 또는 이름")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--minutes", type=float, default=0)
    p = sub.add_parser("match")
    p.add_argument("--pcap", type=Path, required=True)
    p.add_argument("--logs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    {"certs": lambda: make_certs(args.out), "serve": lambda: serve(args.certs, args.bind), "run": lambda: run(args),
     "capture": lambda: capture(args), "match": lambda: match_main(args)}[args.cmd]()


if __name__ == "__main__":
    main()
