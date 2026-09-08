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

## Z-stacks

`POST /v1/zstack` runs FLIMKit's z-stack fit over a folder of slices, one file per slice named `region_z1.ptu`, `region_z2.ptu` and so on. A folder holding several regions is fitted as several stacks in one run.

```
POST /v1/zstack/scan   {"ptu_dir": "/path/slices"}
POST /v1/zstack        {"ptu_dir": "/path/slices", "params": {"n_exp": 2, "z_step_um": 2.0}}
GET  /v1/zstack/defaults
POST /v1/zstack/export {"group_dir": "/path/out/RegionA", "format": "ome-tiff"}
```

The scan is there so a client can say what it found before offering any settings, and refuse an empty folder without starting a job. The fit itself runs as a job, so `GET /v1/jobs/{id}` reports progress by slice and `DELETE` cancels at the next one.

Each stack is fitted as one FOV: the decay is pooled over every slice, the lifetimes are fitted once from the pooled decay and locked, and each slice then gets a per-pixel fit with only the amplitudes free. A single slice rarely carries the photons to identify two lifetimes; the stack usually does.

Each stack comes back as an OME-Zarr store, a `(C, Z, Y, X)` volume with a channel per map: intensity in photons, `tau_mean_int` and `tau_mean_amp` in ns, an `alpha_N` per component, and the bound fraction and chi-squared maps when they were computed. `zarr` and `ome-zarr` are dependencies of this package, not an extra, so the store is always what a client gets unless it asks for something else. `ome-tiff` is that something else, same channels and same axes, for a viewer that will not read the store; `/v1/zstack/export` rewrites a finished run in either format without fitting it again, which is also how to correct a z step entered wrong.

The store carries one resolution level on purpose. A pyramid is built by interpolating between neighbours, and a lifetime map is mostly NaN where nothing was fitted, so the coarse levels a viewer shows when zoomed out would smear that NaN over the pixels that did fit. Chunking, one chunk per slice per channel, is what keeps a large volume readable.

FLIMKit's own per-slice `.npy` maps, `_zseries.csv` and `_zseries.json` are left where it writes them, and the reply names them. Its per-slice PNGs and detail plots are off by default here, since each one costs a second summed fit per slice; `save_plots` turns them back on.

### When the client cannot see the disk

The reply names a path, so a client on the same machine opens the store itself and nothing large goes over HTTP. Over a forwarded port that path means nothing locally, and for that there is:

```
GET /v1/zstack/volume.ome.tif?group_dir=/path/out/RegionA&z_step_um=2.0
```

which builds the volume from the same slice maps and streams it as one OME-TIFF, in blocks out of a temporary file rather than through one bytes object, since a fitted stack is routinely larger than memory. The temporary file goes whether the transfer finished or not. Shape, channels, units and voxel size ride on `X-FLIMKit-Volume-*` headers, and the store on the server is left alone.

The slices themselves are still a server-side path: `ptu_dir` is opened by FLIMKit, not uploaded. A forwarded client picking a local folder gets a 404 naming the folder, before any job starts.

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
