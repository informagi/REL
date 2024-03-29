# REL server

This section describes how to set up and use the REL server.

## Running the server

The server uses [fastapi](https://fastapi.tiangolo.com/) as the web framework. 
FastAPI is a modern, fast (high-performance), web framework for building APIs bases on standard Python type hints. 
When combined with [pydantic](https://docs.pydantic.dev/) this makes it very straightforward to set up a web API with minimal coding.

```bash
python ./src/REL/server.py \
    $REL_BASE_URL \
    wiki_2019 \
    --ner-model ner-fast ner-fast-with-lowercase
```

This will open the API at the default `host`/`port`: <http://localhost:5555>.

One of the advantage of using fastapi is its automated docs by adding `/docs` or `/redoc` to the end of the url:

- <http://localhost:5555/docs>
- <http://localhost:5555/redoc>

You can use `python ./src/scripts/test_server.py` for some examples of the queries and to test the server.

### Setup

Set `$REL_BASE_URL` to the path where your data are stored (`base_url`).

For mention detection and entity linking, the `base_url` must contain all the files specified [here](../how_to_get_started).

In addition, for conversational entity linking, additonal files are needed as specified [here](../conversations)

In summary, these paths must exist:

 - `$REL_BASE_URL/wiki_2019` or `$REL_BASE_URL/wiki_2014`
 - `$REL_BASE_URL/bert_conv`  for conversational EL)
 - `$REL_BASE_URL/s2e_ast_onto ` for conversational EL)

## Running REL as a systemd service

In this tutorial we provide some instructions on how to run REL as a systemd
service. This is a fairly simple setup, and allows for e.g. automatic restarts
after crashes or machine reboots.

### Create `rel.service`

For a basic systemd service file for REL, put the following content into
`/etc/systemd/system/rel.service`:

```ini
[Unit]
Description=My REL service

[Service]
Type=simple
ExecStart=/bin/bash -c "python src/REL/server.py"
Restart=always

[Install]
WantedBy=multi-user.target
```

Note that you may have to alter the code in `server.py` to reflect
necessary address/port changes.

This is the simplest way to write a service file for REL; it could be more
complicated depending on any additional needs you may have. For further
instructions, see e.g. [here](https://wiki.debian.org/systemd/Services) or `man
5 systemd.service`.

### Enable the service

In order to enable the service, run the following commands in your shell:

```bash
systemctl daemon-reload

# For systemd >= 220:
systemctl enable --now rel.service

# For earlier versions:
systemctl enable rel.service
reboot
```
