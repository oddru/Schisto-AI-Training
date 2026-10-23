# Chapter 2 Feature Parameter Summary

*Synthetic dataset assumptions; `Snail_Habitat_Risk` and `AWD_Gate_Action` are excluded labels.*

| Feature             | Unit           | Domain            | Minimum   | Maximum   | Mean    | SD      | AWD / O. hupensis safety logic                                                                              |
|:--------------------|:---------------|:------------------|:----------|:----------|:--------|:--------|:------------------------------------------------------------------------------------------------------------|
| Month               | calendar month | Context           | -         | -         | -       | -       | Seasonal context; wet-season months use the higher synthetic rainfall factor.                               |
| Day of week         | day name       | Context           | -         | -         | -       | -       | Temporal context; no direct ecological threshold is assigned.                                               |
| Hour                | h              | Context           | 0.0000    | 23.0000   | 11.5000 | 6.9230  | Diurnal context for the synthetic sensor stream.                                                            |
| Crop growth stage   | stage          | Agronomic context | -         | -         | -       | -       | Flowering safety takes precedence: water remains above 0 cm and the gate opens below 2 cm.                  |
| Water-table depth   | cm             | Hydrology         | -16.8111  | 5.3740    | -4.1356 | 5.9204  | AWD variable; outside flowering the gate opens at <= -15 cm, while flowering maintains flooding above 0 cm. |
| Inundation duration | days           | Hydrology         | 0.0000    | 15.5833   | 1.1162  | 2.6467  | Simulated O. hupensis habitat duration; >14 days contributes to High risk and >3 days to Moderate risk.     |
| Water-flow velocity | m/s            | Hydrology         | 0.0000    | 0.3497    | 0.1480  | 0.0522  | Hydraulic context; no independent simulated snail-risk threshold is configured.                             |
| Water pH            | pH             | Water quality     | 6.0278    | 7.4046    | 6.7497  | 0.1905  | Simulated habitat risk uses 5.5-7.9 for High risk and 5.3-8.1 for Moderate risk.                            |
| Soil moisture       | % VWC          | Soil              | 20.6836   | 80.9595   | 53.7319 | 14.3704 | Outside flowering, AWD opens at <=25%; habitat-risk conditions require 20-80%.                              |
| Soil temperature    | deg C          | Soil/ecology      | 19.8188   | 32.9970   | 25.4266 | 3.2464  | Simulated habitat risk uses 15-30 deg C for High risk and 15-32 deg C for Moderate risk.                    |
| Vegetation coverage | %              | Vegetation        | 10.0000   | 95.0000   | 52.6084 | 25.9723 | Crop-canopy context; no independent threshold is configured.                                                |
| Vegetation height   | cm             | Vegetation        | 5.0000    | 120.0000  | 62.7648 | 36.1023 | Crop-growth context; no independent threshold is configured.                                                |
