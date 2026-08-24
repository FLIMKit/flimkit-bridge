# flimkit-bridge

The local HTTP server that [FLIMKit](https://github.com/FLIMKit/FLIMKit) image analysis clients talk to.

Two add-ons use it. The [QuPath extension](https://github.com/FLIMKit/flimkit-qupath-bridge) and the [Fiji add-on](https://github.com/FLIMKit/flimkit-fiji-bridge) are both clients of this one server, so a fix here reaches both and neither carries its own copy of the API.

It started inside the QuPath add-on and moved out once there were two clients. The wire protocol it speaks was designed and first implemented in [flimkit-fiji-bridge](https://github.com/FLIMKit/flimkit-fiji-bridge) by Zhen Yuan Yeo (https://doi.org/10.5281/zenodo.21951612).

## Running it

The server starts with FLIMKit. `Tools > FLIMKit Bridge...` shows its address and whether anything has connected.

It also runs without the desktop app, which is what you want on a headless machine or when QuPath or Fiji is the only front end:

```bash
pip install flimkit-bridge
flimkit-bridge
```

`--port` and `--token` are there if you need them, `--force` takes over from a bridge that has been left running, and `--no-announce` serves alongside one instead.

## Pairing

The server writes its address and a generated token to `~/.flimkit/bridge.json`, and clients read that file, so there is nothing to type in. Each start mints a new token, so a client re-reads the file before every call.

It writes `~/.flimkit/qupath-bridge.json` alongside it, carrying the same address under the older protocol name. QuPath extensions built before the server moved out check for that name and stop pairing without it. The second file goes when those versions are retired.

## Versions

`GET /v1/status` reports `protocol_version`, `bridge_version` and `flimkit_version`. A client checks `protocol_version`, which is what governs whether the two can talk. `bridge_version` is for display: the server and each client version independently now, so a difference between them is ordinary rather than a problem.

## Security

The server listens on `127.0.0.1` only, and refuses any request whose `Host` header is not localhost, which stops a web page reaching it by resolving its own hostname to your machine. Every endpoint except the status check requires the token.

Both programs therefore have to be on the same machine. If they are not, forward the port over SSH rather than exposing it:

```bash
ssh -L 8765:127.0.0.1:8765 you@the-flimkit-machine
```

## Licence

MIT. See [LICENSE.md](LICENSE.md).
