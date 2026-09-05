




-- Daily

-- 1. Number of contacts/calls each week for B2C NA

CREATE OR REPLACE TEMPORARY TABLE `contacts_daily` AS

WITH Presented AS (
SELECT
  Week
  ,PresentedDate
  ,'NA Phone' AS Geo
,SUM(Actuals) Actuals
  ,SUM(Abandons) Abandons
FROM (
    SELECT
        DATE_TRUNC(CAST(intervaldate AS DATE), WEEK) Week
        ,CAST(qic.intervaldate AS DATE) PresentedDate
        ,r.reportname
        , SUM(qic.presented) AS Actuals
        , SUM(qic.abandoned) AS Abandons
    FROM `wf-gcp-us-ae-ops-prod.wfm_staging.tbl_queue_interval` qic -- Table wf-gcp-us-ae-ops-prod:wfm_reporting.tbl_queue_interval was not found in location US
    INNER JOIN `wf-gcp-us-ae-ops-prod.wfm_staging.tbl_queue_group_map` qgm ON qic.queueid = CAST(qgm.queueid AS INT64) -- Table wf-gcp-us-ae-ops-prod:wfm_reporting.tbl_queue_group_map was not found in location US
    AND qic.ProdRouterID = qgm.ProdRouterID
    INNER JOIN `wf-gcp-us-ae-ops-prod.wfm_staging.tbl_report_queue_group_map` rqgm ON qgm.queuegroupid = rqgm.queuegroupid -- Table wf-gcp-us-ae-ops-prod:wfm_reporting.tbl_report_queue_group_map was not found in location US
    INNER JOIN `wf-gcp-us-ae-ops-prod.wfm_staging.tbl_report` r ON rqgm.reportid = r.reportid -- Table wf-gcp-us-ae-ops-prod:wfm_reporting.tbl_report was not found in location US
    WHERE
    r.reportname = 'CCPR_NAService'
    AND qic.intervaldate >= '2018-01-07'
    AND qic.intervaldate < CURRENT_DATE()
    GROUP BY 1,2,3
) a
GROUP BY 1,2,3
),

CallBacks AS (
SELECT
  DATE_SUB(CAST(CallDate AS DATE), INTERVAL (EXTRACT(DAYOFWEEK FROM CallDate) - 1) DAY) Week
  ,CallDate
  ,CASE ReportName
     WHEN 'NA Frontline' THEN 'NA Phone'
     WHEN 'UK Frontline' THEN 'UK Phone'
     WHEN 'DE Frontline' THEN 'DE Phone'
     WHEN 'EU Frontline' THEN 'EU Phone'
     ELSE 'NA Specialized'
   END AS Geo
   ,SUM(CallBacks) CallBacks
FROM `wf-gcp-us-ae-ops-prod.wfm_staging.tbl_frontline_call_back_counts` -- Table wf-gcp-us-ae-ops-prod:wfm_reporting.tbl_frontline_call_back_counts was not found in location US
WHERE ReportName = 'NA Frontline'
GROUP BY 1,2,3
),

b2bvol AS (
SELECT
DATE_TRUNC(CAST(r.DateRouted as date),WEEK) Week
, CAST(r.DateRouted as date) as DateRouted
,'NA Phone' AS Geo
, COUNT(DISTINCT ContactID) Contacts
FROM `wf-gcp-us-ae-sql-data-prod.elt_oms.tbl_service_routing_details` r
left join `wf-gcp-us-ae-sql-data-prod.elt_order.tbl_order` od on r.MatchedOrderID=od.OrID
left join `wf-gcp-us-ae-sql-data-prod.elt_order.tbl_order_product` op on r.MatchedOrderID=op.OpOrID
left join `wf-gcp-us-ae-sql-data-prod.elt_product.tblpl_ship_speed` ss on op.OpSpID = ss.SpID
where
SourceID = 2 and routingId = 6 -- Volume that was previously skill 2 and is now b2b
and CAST(r.DateRouted as date) >= '2018-01-07'
and CAST(r.DateRouted as date) <= '2021-09-22' -- Date the B2B unmanaged calls stopped being routed to Service
GROUP BY 1,2
)

SELECT
    p.Week
    ,p.PresentedDate
    ,p.Geo
    ,IFNULL(p.Actuals, 0) Presented
    ,IFNULL(p.Abandons,0) Abandons
    ,IFNULL(CallBacks, 0) AS CallBacks
    ,IFNULL(b.Contacts,0) AS B2B_Contacts
    ,IFNULL(p.Actuals,0) - IFNULL(cb.CallBacks,0) - IFNULL(b.Contacts,0) AS CB_Adj_Presented
FROM Presented p
LEFT JOIN CallBacks cb ON p.Week = cb.Week AND p.Geo = cb.Geo AND p.PresentedDate = cb.CallDate
LEFT JOIN b2bvol b ON p.Week = b.Week AND p.Geo = b.Geo AND p.PresentedDate = b.DateRouted
ORDER BY 1,2,3
;

CREATE OR REPLACE TEMPORARY TABLE `contacts` AS
select
    Week AS actualWeek,
    PresentedDate as PresentedDate,
    CASE WHEN Week <= '2018-08-26' THEN ROUND(sum(CB_Adj_Presented)*0.97) ELSE sum(CB_Adj_Presented) END AS presentedVolume
from `contacts_daily`
where Geo = 'NA Phone'
--Remove unmanaged B2B calls from Service for as long data is available (since 2018-08-26) and for before that,
--decrease it by a fixed 3%.
group by 1, 2
order by 1, 2;


CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpContactsDailyNa AS
select * from contacts order by actualWeek, PresentedDate;

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpContactsDailyNa;






# 2. New order query
CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrders AS
SELECT
    DATE_ADD(DATE(OrCompleteDate), INTERVAL (1 - EXTRACT(DAYOFWEEK FROM DATE(OrCompleteDate))) DAY) as actualWeek
    ,DATE(OrCompleteDate) as actualDate
    ,EXTRACT(DAYOFWEEK FROM DATE(OrCompleteDate)) as DayOfWeek
    ,COUNT(DISTINCT OrID) as numOrders
FROM `wf-gcp-us-ae-sql-data-prod.elt_order.tbl_order` ord
LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_product.tbl_store` st
    ON st.SoID = ord.OrSoID
WHERE
    DATE(OrCompleteDate) >= '2018-01-07'
    AND DATE(OrCompleteDate) < CURRENT_DATE()
    AND st.SoStyID IN (1,6) -- US and CA
    AND ord.OrIsB2BOrder is not true
GROUP BY
    DATE_ADD(DATE(OrCompleteDate), INTERVAL (1 - EXTRACT(DAYOFWEEK FROM DATE(OrCompleteDate))) DAY)
    ,DATE(OrCompleteDate)
    ,EXTRACT(DAYOFWEEK FROM DATE(OrCompleteDate));

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrders order by actualWeek, actualDate;




# frac new customer orders
-- 2.1 fracnewcustomerordersshift
CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrdersCustomerTypeNaB2c AS
SELECT
    DATE(Event_Timestamp) as actualDate
    ,EXTRACT(DAYOFWEEK FROM DATE(Event_Timestamp)) as DayOfWeek
    ,CASE WHEN VisitorType IN ('Activated Customer') THEN 0 ELSE 1 END AS NewCustomer
    -- ,CASE WHEN st.SoStyID = 1 THEN 'US' ELSE 'CA' END AS country
    -- ,CASE WHEN cl.B2BLevel IN ('Basic', 'Premium') THEN 'b2b' ELSE 'b2c' END AS businessType
    ,COUNT(DISTINCT OrID) AS NumOrders
FROM `wf-gcp-us-ae-sf-prod.curated_clickstream.tbl_dash_clicks` cl
LEFT JOIN `wf-gcp-us-ae-sql-data-prod.elt_product.tbl_store` st
    ON st.SoID = cl.Event_SoID
WHERE cl.OrID IS NOT NULL
    AND DATE(Event_Timestamp) >= '2018-01-07'
    AND DATE(Event_Timestamp) < CURRENT_DATE()
    AND st.SoStyID IN (1, 6) -- CA and US
    -- AND cl.B2BLevel IN ('None', 'Basic', 'Premium')
    AND cl.B2BLevel IN ('None') -- b2c only
GROUP BY
    DATE(Event_Timestamp)
    ,EXTRACT(DAYOFWEEK FROM DATE(Event_Timestamp))
    ,CASE WHEN VisitorType IN ('Activated Customer') THEN 0 ELSE 1 END
;


select * from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrdersCustomerTypeNaB2c
order by actualDate, DayOfWeek
limit 1000;



CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpRollingOrdersNaB2c AS
    SELECT
        actualDate
        -- , country
        -- , businessType
        , SUM(NumOrders) AS NumOrdersTotal
    FROM wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrdersCustomerTypeNaB2c
    GROUP BY 1;

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpRollingOrdersNaB2c order by actualDate;


CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpRollingNewCustomerOrdersNaB2c AS
SELECT
    actualDate
    -- ,country
    -- ,businessType
    ,SUM(NumOrders) as numOrdersNewCustomer
FROM wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrdersCustomerTypeNaB2c
WHERE NewCustomer = 1
GROUP BY 1;

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpRollingNewCustomerOrdersNaB2c order by actualDate;

CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpCvtBaseNaB2c AS
SELECT
    t.actualDate
    -- ,t.country
    -- ,t.businessType
    ,NumOrdersTotal
    ,numOrdersNewCustomer
    ,numOrdersNewCustomer/NumOrdersTotal as fracNewCustomerOrders
FROM wf-gcp-us-ae-dsservice-prod.junk.tmpRollingOrdersNaB2c t
JOIN wf-gcp-us-ae-dsservice-prod.junk.tmpRollingNewCustomerOrdersNaB2c n
    ON t.actualDate = n.actualDate
    -- AND t.country = n.country
    -- AND t.businessType = n.businessType
    ORDER BY 1;


select * from wf-gcp-us-ae-dsservice-prod.junk.tmpCvtBaseNaB2c order by 1;

CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpCvtNaB2c AS
select actualDate,	fracNewCustomerOrders
from wf-gcp-us-ae-dsservice-prod.junk.tmpCvtBaseNaB2c;

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpCvtNaB2c order by actualDate;




















CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrdersNaB2c AS
select v.actualWeek, v.presentedDate, v.presentedVolume,
tto.numOrders as numOrdersTotal,
nco.fracNewCustomerOrders as fracNewCustomerOrders,
from wf-gcp-us-ae-dsservice-prod.junk.tmpContactsDailyNa v
left join wf-gcp-us-ae-dsservice-prod.junk.tmpDailyOrders tto
on date(v.presentedDate) = tto.actualDate
left join wf-gcp-us-ae-dsservice-prod.junk.tmpCvtNaB2c nco
on date(v.presentedDate) = nco.actualDate
where v.presentedDate >= '2018-01-07'
order by v.presentedDate
;

select * from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrdersNaB2c;







# 3. Holidays
CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.usHolidays AS
SELECT distinct ed.edeventdate as date, em.emeventcountry as country, em.emeventname as name
FROM `wf-gcp-us-ae-ops-prod.wfm_reporting.tbl_event_dates` ed
LEFT JOIN `wf-gcp-us-ae-ops-prod.wfm_reporting.tbl_event_map` em on ed.edEventID = em.emEventID
WHERE em.emEventCountry IN ('US') and ed.edeventdate >= '2016-01-01'
order by ed.edeventdate asc;

CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.caHolidays AS
SELECT distinct ed.edeventdate as date, em.emeventcountry as country, em.emeventname as name
FROM `wf-gcp-us-ae-ops-prod.wfm_reporting.tbl_event_dates` ed
LEFT JOIN `wf-gcp-us-ae-ops-prod.wfm_reporting.tbl_event_map` em on ed.edEventID = em.emEventID
WHERE em.emEventCountry IN ('CA') and ed.edeventdate >= '2016-01-01'
order by ed.edeventdate asc;




# Join volume, orders and holidays
CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrderHolidaysNaB2c AS
select t.actualWeek,
IFNULL(IFNULL(t.presentedDate, ush.date), cah.date) as presentedDate,
t.presentedVolume, t.numOrdersTotal, t.fracNewCustomerOrders,
ush.name as usHoliday, cah.name as canadaHoliday
from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrdersNaB2c t
full outer join wf-gcp-us-ae-dsservice-prod.junk.usHolidays ush
on t.presentedDate = ush.date
full outer join wf-gcp-us-ae-dsservice-prod.junk.caHolidays cah
on IFNULL(t.presentedDate, ush.date) = cah.date
order by presentedDate
;

CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrderHolidaysNaB2c AS
select
DATE_SUB(CAST(presentedDate AS DATE), INTERVAL (EXTRACT(DAYOFWEEK FROM presentedDate) - 1) DAY) actualWeek,
presentedDate, presentedVolume, numOrdersTotal, fracNewCustomerOrders,
usHoliday, canadaHoliday
from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrderHolidaysNaB2c
order by actualWeek, presentedDate;

CREATE OR REPLACE TABLE wf-gcp-us-ae-dsservice-prod.junk.dailyContactsOrderHolidaysNaB2cV3 AS
select * from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrderHolidaysNaB2c
where actualWeek < DATE_SUB(CAST(CURRENT_DATE() AS DATE), INTERVAL (EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) - 15) DAY)
and actualWeek > '2018-01-01'
order by actualWeek, presentedDate;

select * from wf-gcp-us-ae-dsservice-prod.junk.dailyContactsOrderHolidaysNaB2cV3
order by actualWeek, presentedDate;

-- TEST OF PARTIAL WEEK
-- select * from wf-gcp-us-ae-dsservice-prod.junk.tmpDailyContactsOrderHolidaysNaB2c
-- where actualWeek <= DATE_SUB(CAST(date('2022-05-05') AS DATE), INTERVAL (EXTRACT(DAYOFWEEK FROM date('2022-05-05')) - 8) DAY)
-- order by actualWeek, presentedDate;