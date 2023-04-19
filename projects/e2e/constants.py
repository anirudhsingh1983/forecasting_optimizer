
# data attributes
isc_sls = ['ACN', 'ACI', 'NVO', 'BCO', 'DRA']
al_sls = ['ACN', 'ACI']
non_al_sls = list(set(isc_sls) - set(al_sls))
nrv_sls = ['NRV']

SPOID_COL = 'SpoID'
SERVICELEVEL_COL = 'ServiceLevel'
ACF_LOCATION_NAME_COL = 'CFSLocation'
ORIGIN_PORT_NAME_COL = 'Origin'
DESTINATION_PORT_NAME_COL = 'Destination'
FC_NAME_COL = 'DestinationWarehouse'
FC_ID_COL = 'WarehouseId'
QUANTITY_COL = 'SpiQty'

ACF_ARRIVAL_TIMESTAMP = 'ActualDeliveryCFS'
ACF_DEPARTURE_TIMESTAMP = 'ActualDepartureCFS'
ORIGIN_PORT_ARRIVAL_TIMESTAMP = 'GateInOriginPort'
ORIGIN_PORT_DEPARTURE_TIMESTAMP = 'DepartureOriginPort'
DESTINATION_PORT_ARRIVAL_TIMESTAMP = 'ArrivalDestinationPort'
DESTINATION_PORT_DEPARTURE_TIMESTAMP = 'GateOutDestinationPort'
FC_ARRIVAL_TIMESTAMP = 'CompletedTime'


# features and targets
DATE_STR = 'date'
DAYOFMONTH_STR = 'day'
DAYOFWEEK_STR = 'dayofweek'
WEEK_STR = 'week'
MONTH_STR = 'month'

NODE_NAME_STR = 'node_name'
CSF_TO_ORIGIN_PORT_STR = 'csf_to_origin_port'
ORIGIN_PORT_TO_DESTINATION_PORT_STR = 'origin_port_to_destination_port'
DESTINATION_PORT_TO_FC_STR = 'destination_port_to_fc'
STAY_AT_CSF_STR = 'stay_at_csf'
STAY_AT_ORIGIN_PORT_STR = 'stay_at_origin_port'
STAY_AT_DESTINATION_PORT_STR = 'stay_at_destination_port'
ACTUAL_DISTRIBUTION_STR = 'actual_dist'
LENGTH_STR = 'len'
CLASSIFICATION_STR = 'classification'
PROBABILITY_STR = 'probability'
LEFT_STR = 'left'
OBJECT_STR = 'object'
INBOUND_FORECAST_STR = 'inbound_forecast'
NO_CFS_STR = 'no_asia_consolidation'

# Models
PREV_NODE_VOL_MODEL = 'prev_node_vol_model'
TRANSIT_TIME_MODEL = 'transit_time_model'
DWELL_TIME_MODEL = 'dwell_time_model'
MODELS = [PREV_NODE_VOL_MODEL, TRANSIT_TIME_MODEL, DWELL_TIME_MODEL]

# misc
ARRIVAL_STR = 'arrival'
DEPARTURE_STR = 'departure'
SERVICELEVEL_STR = 'ServiceLevel'
MODEL_STR = 'model'
EVALUATION_METRICS = 'evaluation_metrics'
JS_DIVERGENCE_STR = 'jsd'
FEATURE_IMPORTANCE_STR = 'feature_importance'
SOURCE_STR = 'source'
DESTINATION_STR = 'destination'

FC_KEY = FC_ID_COL

ISC_FRACTION = 0.6

# paths
LOCAL_DATA_DIR = 'tmp'

# forecast output
FORECAST_PUBLISH_ID = 'publish_date'
CGF_SERVICELEVEL_STR = 'ISC'
E2E_FORECAST_COL = 'forecast'
LEG_COL = 'leg'
FACILITY_NAME_COL = 'facility_name'