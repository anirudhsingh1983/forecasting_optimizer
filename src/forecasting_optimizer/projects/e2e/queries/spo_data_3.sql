
-- This foundational table query is provided by SCDE
CREATE OR REPLACE TEMPORARY TABLE cgf_temp AS
  SELECT
    SpoID
    ,MAX(CFSLocation) AS CFSLocation
    ,MAX(Origin) AS OriginPortCode
    ,MAX(Destination) AS DestinationPortCode
    ,MIN(ActualDeliveryCFS) AS ActualDeliveryCFS
    ,MAX(ActualDepartureCFS) AS ActualDepartureCFS
    ,MAX(GateInOriginPort) AS GateInOriginPort
    ,MAX(DepartureOriginPort) AS DepartureOriginPort
    ,MAX(ArrivalDestinationPort) AS ArrivalDestinationPort
    ,MAX(GateOutDestinationPort) AS GateOutDestinationPort
FROM `wf-gcp-us-ae-ops-prod.csn_reporting_isc.datamart_combined`
GROUP BY 1
;

CREATE OR REPLACE TEMPORARY TABLE arrivals_temp_aggregated AS
SELECT
    SpoID
    ,MIN(ArrivalDate) AS ArrivalDate
    ,SUM(spiqty) AS SpiQty
FROM `wf-gcp-us-ae-ops-prod.buyfair_adhoc.tbl_arrivals_report`
WHERE forecast_element = "ISC"
GROUP BY 1
;

CREATE OR REPLACE TEMPORARY TABLE non_arrival_temp AS
SELECT
  b.spoid as SPOID
  ,CASE
    WHEN SpiSpoReasonId IN (2,3,4,10,12,33,34,36,52,53,13,24,26,28,30,38) AND  SpoUseWFAccount = true
    THEN 'Freight Collect Stocking Domestic'
    WHEN SpiSpoReasonId IN (2,3,4,10,12,33,34,36,52,53,13,24,26,28,30,38) AND SpoUseWFAccount = false
    THEN 'Non Freight Collect Stocking Domestic'
    WHEN SpiSpoReasonId IN (40) AND  SpoUseWFAccount = true
    THEN 'Freight Collect CG Domestic'
    WHEN SpiSpoReasonId IN (40) AND  SpoUseWFAccount = false
    THEN 'Non Freight Collect CG Domestic'
    WHEN SpiSpoReasonId IN (23)
    THEN 'Warehouse Transfer'
    ELSE i.IslCode
  END AS servicelevel
  ,b.spowhid as Original_WH
  ,wh.wswhname as Original_WH_Name
  ,cgf.CFSLocation
  ,cgf.OriginPortCode
  ,cgf.DestinationPortCode
  ,spocreated as DateEntered
  ,cgf.ActualDeliveryCFS
  ,cgf.ActualDepartureCFS
  ,cgf.GateInOriginPort
  ,cgf.DepartureOriginPort
  ,cgf.ArrivalDestinationPort
  ,cgf.GateOutDestinationPort
  ,wh.wsisbreakbulkfacility
  ,min(aa.arrivaldate) as arrivaldate
  ,COALESCE(MAX(aag.spiqty),sum(a.spiqty)) as Total_units
  ,sum(
      CASE WHEN pt.productcartontypeid = 1
      THEN a.spiqty
      ELSE 0 end
      ) as Type_1_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 2
      THEN a.spiqty
      ELSE 0 end
      ) as Type_2_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 3
      THEN a.spiqty
      ELSE 0 end
      ) as Type_3_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 4
      THEN a.spiqty
      ELSE 0 end
      ) as Type_4_Units
  ,sum(
      CASE WHEN pt.productcartontypeid is null
      THEN a.spiqty
      ELSE 0 end
      ) as No_Type_Units
FROM
  `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_purchase_order_item` a
  INNER JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_purchase_order` b ON a.spispoid = b.spoid
  LEFT JOIN  `wf-gcp-us-ae-sql-data-prod.csn_wms.tblpl_inbound_service_level` i ON b.SpoInboundServiceLevelID = i.IslID
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tblpl_stock_purchase_order_reason_code` d ON a.spisporeasonid = d.sporeasonid
  LEFT JOIN  `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_product` spr ON a.spisprid = spr.sprid
  LEFT JOIN   `wf-gcp-us-ae-sql-data-prod.elt_datagov.tbl_join_manufacturer_part_product_carton_type` pt  ON pt.ManufacturerPartId = spr.SprManufacturerPartID
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_wms_supplier` wh ON spowhid = wswhid
  INNER JOIN (SELECT spoid, min(arrivaldate) arrivaldate
           FROM `wf-gcp-us-ae-ops-prod.buyfair_adhoc.tbl_arrivals_report`
           WHERE forecast_element = "ISC"
           GROUP BY spoid
           )aa ON aa.spoid = b.spoid
  LEFT JOIN `wf-gcp-us-ae-bulk-prod.csn_wms.tblpl_stock_purchase_order_status` ON SpoStatus = spostatusid
  LEFT JOIN cgf_temp cgf ON b.spoid = cgf.SpoID
  LEFT JOIN arrivals_temp_aggregated aag ON b.SpoID = aag.SpoID
WHERE
  1=1
  AND b.spoid = b.spoparentspoid --only parent spos
  AND b.SpoIsCancelled = false
  AND (
      i.islcode <> 'ACI'
      or i.islcode is null
      )
  AND spocreated > date_add(current_date(), INTERVAL -2 year)
  AND wsstyid in (1,6)
  AND (
      wsiscgspoenabled=true
      or wsisbreakbulkfacility=true
      )
  AND (
      spisporeasonid in (2,3,4,10,12,33,34,36,52,53,13,24,26,28,30,38,40,23) #Relevant SPOs for CG inbound
      or islcode is not null #all ISC SPOs
      )
  AND a.spiqty >0
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
;

CREATE OR REPLACE TEMPORARY TABLE arrivals_temp AS
SELECT
  arrivals.spoid as SPOID
  ,CASE
    WHEN SpiSpoReasonId IN (2,3,4,10,12,33,34,36,52,53,13,24,26,28,30,38) AND  SpoUseWFAccount = true
    THEN 'Freight Collect Stocking Domestic'
    WHEN SpiSpoReasonId IN (2,3,4,10,12,33,34,36,52,53,13,24,26,28,30,38) AND SpoUseWFAccount = false
    THEN 'Non Freight Collect Stocking Domestic'
    WHEN SpiSpoReasonId IN (40) AND  SpoUseWFAccount = true
    THEN 'Freight Collect CG Domestic'
    WHEN SpiSpoReasonId IN (40) AND  SpoUseWFAccount = false
    THEN 'Non Freight Collect CG Domestic'
    WHEN SpiSpoReasonId IN (23)
    THEN 'Warehouse Transfer'
    ELSE i.IslCode
  END AS servicelevel
  ,b.spowhid as Original_WH
  ,wh.wswhname as Original_WH_Name
  ,cgf.CFSLocation
  ,cgf.OriginPortCode
  ,cgf.DestinationPortCode
  ,spocreated as DateEntered
  ,cgf.ActualDeliveryCFS
  ,cgf.ActualDepartureCFS
  ,cgf.GateInOriginPort
  ,cgf.DepartureOriginPort
  ,cgf.ArrivalDestinationPort
  ,cgf.GateOutDestinationPort
  ,wh.wsisbreakbulkfacility
  ,min(arrivals.arrivaldate) as arrivaldate
  ,MAX(arrivals.spiqty) as Total_units
  ,sum(
      CASE WHEN pt.productcartontypeid = 1
      THEN a.spiqty
      ELSE 0 end
      ) as Type_1_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 2
      THEN a.spiqty
      ELSE 0 end
      ) as Type_2_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 3
      THEN a.spiqty
      ELSE 0 end
      ) as Type_3_Units
  ,sum(
      CASE WHEN pt.productcartontypeid = 4
      THEN a.spiqty
      ELSE 0 end
      ) as Type_4_Units
  ,sum(
      CASE WHEN pt.productcartontypeid is null
      THEN a.spiqty
      ELSE 0 end
      ) as No_Type_Units
FROM
  arrivals_temp_aggregated arrivals
  LEFT JOIN non_arrival_temp t ON arrivals.SpoID = t.SpoID
  LEFT JOIN cgf_temp cgf ON arrivals.spoid = cgf.SpoID
  INNER JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_purchase_order_item` a ON arrivals.spoid = a.spispoid
  INNER JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_purchase_order` b ON a.spispoid = b.spoid
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.csn_wms.tblpl_inbound_service_level` i ON b.SpoInboundServiceLevelID = i.IslID
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tblpl_stock_purchase_order_reason_code` d ON a.spisporeasonid = d.sporeasonid
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_stock_product` spr ON a.spisprid = spr.sprid
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_datagov.tbl_join_manufacturer_part_product_carton_type` pt  ON pt.ManufacturerPartId = spr.SprManufacturerPartID
  LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_wms.tbl_wms_supplier` wh ON b.spowhid = wswhid
  LEFT JOIN `wf-gcp-us-ae-bulk-prod.csn_wms.tblpl_stock_purchase_order_status` ON SpoStatus = spostatusid
WHERE
  t.SpoID IS NULL
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
;

-- final temp table unioning the above two temp tables
SELECT
SPOID,
servicelevel,
Original_WH,
Original_WH_Name,
CFSLocation,
OriginPortCode,
DestinationPortCode,
DateEntered,
ActualDeliveryCFS,
ActualDepartureCFS,
GateInOriginPort,
DepartureOriginPort,
ArrivalDestinationPort,
GateOutDestinationPort,
wsisbreakbulkfacility,
arrivaldate,
Total_units,
Type_1_Units,
Type_2_Units,
Type_3_Units,
Type_4_Units,
No_Type_Units
FROM non_arrival_temp
UNION DISTINCT
SELECT
SPOID,
servicelevel,
Original_WH,
Original_WH_Name,
CFSLocation,
OriginPortCode,
DestinationPortCode,
DateEntered,
ActualDeliveryCFS,
ActualDepartureCFS,
GateInOriginPort,
DepartureOriginPort,
ArrivalDestinationPort,
GateOutDestinationPort,
wsisbreakbulkfacility,
arrivaldate,
Total_units,
Type_1_Units,
Type_2_Units,
Type_3_Units,
Type_4_Units,
No_Type_Units
FROM arrivals_temp atmp
WHERE NOT EXISTS (SELECT natmp.SpoID FROM non_arrival_temp natmp WHERE atmp.SpoID = natmp.SpoID) -- to ensure to union only those SPOs that are not already considered by the earlier logic
;

-- select * from wf-gcp-us-ae-ops-prod.csn_junk.inboud_viz_final_test;

