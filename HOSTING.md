# Hosting

The React frontend runs on Vercel. The Python backend runs on this computer at
`127.0.0.1:8001` and is exposed through a Cloudflare Quick Tunnel.

The frontend reads the current public backend address from
`public/backend.json`. Two user-level systemd services keep the backend and the
tunnel running. When the tunnel restarts, its new address is committed to
GitHub and the Vercel production deployment is updated automatically.

```bash
systemctl --user status finance1-backend finance1-quick-tunnel
journalctl --user -u finance1-quick-tunnel -f
```

The computer must remain awake, connected to the internet, and logged in. A
Quick Tunnel is public and unauthenticated, and its address changes after a
restart.
