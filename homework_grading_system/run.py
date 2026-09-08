"""启动脚本 — python run.py (HTTP服务, 端口8080)"""
import io
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def get_local_ips():
    """获取本机所有可访问地址"""
    hostname = socket.gethostname()
    ips = {"ipv4": [], "ipv6": []}

    try:
        for info in socket.getaddrinfo(hostname, None):
            family, _, _, _, sockaddr = info
            ip = sockaddr[0]
            if family == socket.AF_INET6 and not ip.startswith("fe80:") and "%" not in ip:
                if ip != "::1":
                    ips["ipv6"].append(ip)
            elif family == socket.AF_INET and not ip.startswith("127."):
                ips["ipv4"].append(ip)
    except Exception:
        pass

    return ips


def print_banner(port=8080):
    """打印启动横幅和访问地址"""
    ips = get_local_ips()
    lines = [
        "",
        "  ╔══════════════════════════════════════════╗",
        "  ║     📝 作业批改系统 v4.4                  ║",
        "  ╠══════════════════════════════════════════╣",
        "  ║  本地:  http://localhost:{:<5}            ║".format(port),
        "  ║  本地:  http://127.0.0.1:{:<5}            ║".format(port),
    ]
    for ip in ips["ipv4"]:
        lines.append("  ║  IPv4:  http://{}:{:<5}            ║".format(ip, port))
    for ip in ips["ipv6"]:
        lines.append("  ║  IPv6:  http://[{}]:{:<5}   ║".format(ip, port))
    lines.append("  ╚══════════════════════════════════════════╝")
    lines.append("")
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


def main():
    import uvicorn
    print_banner(8080)
    uvicorn.run("server:app", host="::", port=8080, log_level="info", reload=True)


if __name__ == "__main__":
    main()
