# gods-eye-view: CCTV / ALPR / people audit and disable plan

**Status: PLAN ONLY. Nothing in `vendor/gods-eye-view` has been changed.** Approve this before implementation.

Audited: shallow clone of `harishrajCEO/gods-eye-view` (fork of `bilawalsidhu/gods-eye-view`, MIT, v0.3.0),
`src/`, `server/`, `config/`, `.env.example`. Rule being applied: `docs/ACCEPTABLE_USE.md` (objects only; no CCTV,
no licence plates, no tracking people).

## Findings

| # | Feature | What it does | Where | Verdict |
|---|---|---|---|---|
| 1 | **CCTV layer** (`cctv`) | Pins public traffic/city cameras on the globe and **streams their live frames/video** through the app's own server | client `src/layers/cctv/*` (ingestion, `videoPlayback.js`, `frames.js`), `src/app/layers/cctv.js`, `src/cctvFocusPolicy.js`, `src/cctvFocusRequest.js`; server `server/providers/cctv.js` + `server/providers/cctv/*` (routes `/api/cctv/sources`, `/health`, `/frame/...`, `/media/...`; providers for Lithuania, Taiwan, Ontario 511); source packs `config/cctv_sources.{austin,shinjuku,tallinn,warendorf}.json`; env `CCTV_SOURCES_FILE` | **Disable.** CCTV ingestion is explicitly out of scope. |
| 2 | **ALPR layer** (`alpr`) | Maps the **locations of licence-plate-reader cameras** (OpenStreetMap data via Overpass). It does not read plates itself | `src/layers/alpr/*`, `src/app/layers/alprCameras.js`, `src/data/alprCameras.js` | **Disable.** Not plate reading, but plate-surveillance infrastructure is outside a satellite-labelling product and invites the wrong reading of the brand. |
| 3 | **Street Level layer** (`street-level`) | Mapillary (and optionally Google Street View) ground photos, which contain people, faces and plates (Mapillary blurs them; quality varies) | `src/layers/streetLevel/*`; server `server/providers/mapillary.js` (`/api/mapillary/*`); env `MAPILLARY_CLIENT_TOKEN`, Google Street View via the Google key | **Disable by default.** Street-level imagery of people is outside scope. |
| 4 | Voice/LLM control | Voice agent tool descriptions can drive any layer, including the above | `server/providers/openai/instructions.js`, `toolDescriptions.js` | **Covered** if 1-3 are removed server-side: the tools then have nothing to open. Verify after implementation. |
| 5 | Military awareness / installations | Shows military installations (public data) near a selected aircraft/vessel | `src/layers/awareness/*`, `src/data/militaryAwarenessEngine.js`, `src/app/layers/militaryInstallations.js` | **Not a people issue; your call.** Objects/infrastructure only, but the master context advises civil "monitoring" framing. Suggest off by default in Skyvonyx demos. |
| 6 | "Contacts detection", cockpit "NVG/thermal/surveillance" styles | Visual styles and tactical presets for aircraft/vessel tracks. **No computer-vision or person-detection code was found** (no tfjs/ONNX/YOLO/MediaPipe; "person" hits are "third-person view" etc.) | `src/contactsDetectionPolicy.js`, `src/cockpitVisionPolicy.js` | **No action** for the people rule. Cosmetic naming only. |

Not checked: `node_modules/`, the `build/` output, git history, runtime behaviour. The audit is a static read.

## Proposed change (no upstream history rewrite)

A single flag, `SKYVONYX_OBJECTS_ONLY`, **on by default** in any Skyvonyx build, applied as a **patch file** kept in
this repo (`patches/gods-eye-view-objects-only.patch`) and applied at deploy time with `git apply`. The fork's history
stays as-is; the patch is reviewable and can be re-based when upstream changes.

Defence in depth: hiding a button is not enough, because the server would still proxy camera streams to anyone who
calls `/api/cctv/...`. So the flag acts in three places:

1. **Server (the part that matters):** when the flag is on, do not register the CCTV provider or the Mapillary
   provider at all. `/api/cctv/*` and `/api/mapillary/*` return 404. Ignore `CCTV_SOURCES_FILE`.
2. **Client:** remove `cctv`, `alpr` and `street-level` from the layer catalog (`src/app/catalog.js` and wherever
   layers are mounted - **to be located during implementation**; it was not in `scene.js`/`controls.js`), so they
   never appear in menus, URL state, voice tools or first-run tours.
3. **Config/assets:** in the Skyvonyx deploy, do not ship `config/cctv_sources.*.json`, and leave
   `MAPILLARY_CLIENT_TOKEN` and Google Street View access unset.

Plus a test in the patch: with the flag on, a request to `/api/cctv/sources` is 404 and the catalog contains none of
the three layer ids.

Then, separately: point the GeoJSON layer at the Skyvonyx API's `GET /jobs/{id}/geojson` (approved jobs only).

## Open questions for you
1. Approve disabling 1-3? Off-by-default-but-toggleable, or **removed entirely** in Skyvonyx builds (recommended:
   removed - a toggle invites "just for this demo").
2. Military awareness (5): off by default in Skyvonyx builds, or leave?
3. Keep the patch in this repo (recommended) or commit it on a branch of your fork?
