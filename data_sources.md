### Small-Region Grid Data Sources and Citations (CPAU, Kumo-first)

- **City load (CPAU import/system load)**
  - Source: CPAU internal data export or API
  - Fields: ts (UTC, tz-aware), load_mw, quality_flag
  - License: By data-sharing agreement with CPAU

- **Outages (CPAU OMS)**
  - Source: CPAU OMS export; public info page: [Palo Alto Utilities Outages](https://www.paloalto.gov/Departments/Utilities/Utilities-Services-Safety/Outages)
  - Fields: start_ts, end_ts, cause_code, area/feeder, customers_affected
  - License: By data-sharing agreement with CPAU (OMS); public page terms apply otherwise

- **Weather history**
  - Primary: [Meteostat](https://dev.meteostat.net/) – Python docs: [Hourly](https://dev.meteostat.net/python/hourly.html)
  - Alt: NOAA ISD (NCEI)
  - Fields: ts, temp, dew_point_or_rh, wind_speed, wind_gust, precip
  - License: As per Meteostat/NOAA terms; attribution required

- **Weather forecast + forecast archive (NWS)**
  - API: [NWS API](https://www.weather.gov/documentation/services-web-api)
  - Endpoints: `GET /points/{lat},{lon}` then `GET /gridpoints/{office}/{x},{y}/forecast/hourly`
  - Practice: Archive each issuance with `User-Agent` header per NWS policy; store `ts_issue` and raw forecast JSON
  - License: US Government open data; attribution recommended

- **Grid prices and stress (CAISO)**
  - OASIS API docs: [CAISO OASIS](http://oasis.caiso.com/mrioasis/)
  - LMP: `queryname=PRC_LMP`, `MARKET_RUN_ID=DAM`, node `DLAP_PGAE-APND` (preferred) or `TH_NP15` (fallback); `resultformat=6` (CSV ZIP)
  - Statewide demand: [Today’s Outlook JSON](http://content.caiso.com/outlook/SystemLoad.json)
  - License: CAISO terms; attribution required

- **DER (BTM PV) interconnections**
  - CPUC NEM: [Net Energy Metering](https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/net-energy-metering)
  - Fields: zip_or_feeder, installed_kw, interconnection_date
  - License: Agency/utility terms; attribution required

- **Solar irradiance (for PV proxy)**
  - NSRDB portal: [NSRDB](https://nsrdb.nrel.gov/) – API: [NREL Developer](https://developer.nrel.gov/docs/solar/nsrdb/)
  - Fields: ts, ghi, dni, dhi, cloud_cover
  - License: NREL/NOAA terms; API key required

- **EV charging infrastructure (proxy for EV adoption/usage)**
  - AFDC API: [Alt Fuel Stations](https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/)
  - Params: `fuel_type=ELEC`, `latitude`, `longitude`, `radius` (mi), `api_key`
  - License: DOE/NREL terms; API key required

- **Mobility (traffic counts)**
  - Caltrans PeMS: [Portal](https://pems.dot.ca.gov/) – Docs: [PeMS Docs](https://pems.dot.ca.gov/?dnode=Docs)
  - Notes: Account required; export nearest stations to Palo Alto (US‑101 / I‑280)
  - License: Agency terms; attribution required

- **Calendar and holidays**
  - Python library: [`holidays`](https://pypi.org/project/holidays/)
  - Local school calendars: PAUSD
  - License: Library BSD; check local site terms for calendars

Attribution and license notes are subject to provider updates. Follow each provider’s usage policy and include `User-Agent` headers and API keys where required.


