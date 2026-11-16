# Acceptable use

Skyvonyx labels **objects and land features** in satellite and aerial imagery. That is the whole scope.

## In scope
- Vessels, aircraft, vehicles, buildings, storage tanks, infrastructure, and similar physical objects.
- Land and water features: flood extent, burned area, land cover, change between dates.
- Counting and locating those objects for civil and dual-use **monitoring** (ports, corridors, disasters, environment).

## Out of scope - the code and the company do not do this
- Identifying, recognising, or re-identifying **people** (faces, gait, clothing, or any other means).
- **Tracking individuals** or following a specific person or their vehicle over time.
- **Licence plates** (ALPR/ANPR) or any vehicle-to-owner linkage.
- Ingesting **CCTV**, body-cam, phone, or street-level video feeds.
- Combining our outputs with personal data to profile or target individuals.

Satellite imagery cannot resolve faces or plates anyway; these rules also cover any future data source,
partner integration, or front-end layer.

## Defence and security work
Allowed only as object-level monitoring under the rules above, framed as dual-use monitoring, after incorporation,
and **with legal advice before any contract**. No imagery of restricted or sensitive areas without that advice.

## How this is enforced
- `CLAUDE.md` carries these rules for every AI-assisted code change.
- The `gods-eye-view` front end (a fork) contains CCTV and ALPR layers that must stay disabled in any Skyvonyx
  deployment; see the disable plan in `docs/GODS_EYE_LAYER_AUDIT.md`.
- Every delivery passes the human review gate; a reviewer rejects any job outside this scope.
- A request that falls outside scope is declined, and the decision is recorded.

Owner: founders. Review this document before every new customer category.
