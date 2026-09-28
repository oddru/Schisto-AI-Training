# Thesis Details

## Project Title

**Rice Paddy Alternate Wetting and Drying (AWD) Monitoring and Snail Habitat Risk Management System**

## Project Purpose

The proposed system is intended to help an administrator monitor rice-paddy field conditions and support AWD water-management decisions. It organizes observations from the Philippines rice-paddy dataset into records that can be reviewed for crop-stage context, water availability, soil condition, snail-habitat risk, and required gate action.

The system is a monitoring and decision-support project. It does not claim to replace an agricultural expert, automatically control a physical irrigation gate, or diagnose a crop or human disease.

## Intended Users

- **Administrator** – Maintains the records and reviews summaries and risk-related KPIs.
- **Farm technician or field staff** – Records or verifies field observations.
- **Researcher or adviser** – Reviews trends in field conditions, snail-habitat risk, and AWD actions.

## Primary Data Fields in the Record Table

The following six fields are the proposed primary fields for an individual monitoring record. The first field is a deliberate composite of the dataset's separate `Month`, `Day_of_Week`, and `Hour` columns.

1. **Record Date and Time** (`Month`, `Day_of_Week`, `Hour`) – Identifies when the field observation was recorded. The value may be displayed as one field in the interface while remaining separate source values in the dataset.
2. **Crop Growth Stage** (`Crop_Growth_Stage`) – A categorical value describing the rice production stage, such as transplanting. This provides context for interpreting the field measurements.
3. **Water-Table Depth (cm)** (`Water_Table_Depth_cm`) – A numeric measurement of the water-table depth. It helps the administrator assess the field's water condition for AWD monitoring.
4. **Soil Moisture (VWC %)** (`Soil_Moisture_VWC_Percent`) – A numeric percentage representing the soil's volumetric water content. It provides an additional indicator of field moisture.
5. **Snail Habitat Risk** (`Snail_Habitat_Risk`) – A categorical risk classification, such as Low, Moderate, or High, used to prioritize review and possible intervention.
6. **AWD Gate Action** (`AWD_Gate_Action`) – The recorded or recommended gate-control action associated with the observation. The system should document the meaning of each encoded value, including whether `0` means that no action is required or represents a particular action.

Other dataset measurements, including inundation duration, water-flow velocity, water pH, soil temperature, vegetation coverage, and vegetation height, may be retained as supporting fields or used for later analysis. They are not required to be part of the primary record-table fields.

## Three Key Performance Indicators (KPIs) for the Admin

1. **Total Monitoring Records** – The total count of valid observation records in the system. This indicates the amount of monitoring data available for review.
2. **High Snail-Habitat-Risk Records** – The count of records whose `Snail_Habitat_Risk` value is High. This helps the administrator identify how many observations require priority attention.
3. **AWD Gate Actions Required** – The count of records whose `AWD_Gate_Action` value indicates that an action is needed. The exact calculation must follow the project's documented action-code definitions; the system should not assume that every nonzero value has the same meaning.

## Scope

The proposed project covers recording, viewing, filtering, and summarizing rice-paddy monitoring observations. Its administrative dashboard can present the three KPIs and support review of records by date/time, crop growth stage, water condition, snail-habitat risk, and gate action.

The project does not require unsupported features such as automated irrigation hardware control, medical diagnosis, or real-time sensor integration unless those features are separately approved and implemented as part of the thesis.

## Summary

The selected fields and KPIs connect the dataset to a practical administrative workflow: field staff provide observations, the administrator reviews current conditions and risk levels, and the system summarizes records that may require water-management attention. This keeps the project aligned with the available dataset while leaving the exact interpretation of AWD action codes to be confirmed with the adviser or domain expert.
