# Running REL as a systemd service
In this tutorial we provide some instructions on how to run REL as a systemd
service. This is a fairly simple setup, and allows for e.g. automatic restarts
after crashes or machine reboots.

## rel.service
For a basic systemd service file for REL, put the following content into
`/etc/systemd/system/rel.service`:

```ini
[Unit]
Description=My REL service

[Service]
Type=simple
ExecStart=/bin/bash -c "python -m rel.scripts.code_tutorials.run_server"
Restart=always

[Install]
WantedBy=multi-user.target
```

Note that you may have to alter the code in `run_server.py` to reflect
necessary address/port changes.

This is the simplest way to write a service file for REL; it could be more
complicated depending on any additional needs you may have. For further
instructions, see e.g. [here](https://wiki.debian.org/systemd/Services) or `man
5 systemd.service`.

## Enable the service
In order to enable the service, run the following commands in your shell:

```bash
systemctl daemon-reload

# For systemd >= 220:
systemctl enable --now rel.service

# For earlier versions:
systemctl enable rel.service
reboot
```
