-- ====================================================================
-- Project 5: Industrial Predictive Maintenance - Analytical SQL Queries
-- Demonstrating JOINs, CTEs, Window Functions, and Data Analytics
-- ====================================================================

-- --------------------------------------------------------------------
-- Query 1: Failure rate analysis by Product Type (L, M, H) using CTE
-- --------------------------------------------------------------------
WITH TypeStats AS (
    SELECT 
        Type,
        COUNT(*) AS Total_Units,
        SUM(Machine_Failure) AS Total_Failures,
        SUM(TWF) AS TWF_Count,
        SUM(HDF) AS HDF_Count,
        SUM(PWF) AS PWF_Count,
        SUM(OSF) AS OSF_Count,
        SUM(RNF) AS RNF_Count
    FROM cleaned_sensor_features
    GROUP BY Type
)
SELECT 
    Type,
    Total_Units,
    Total_Failures,
    ROUND(CAST(Total_Failures AS FLOAT) * 100.0 / Total_Units, 2) AS Failure_Rate_Pct,
    TWF_Count, HDF_Count, PWF_Count, OSF_Count, RNF_Count
FROM TypeStats
ORDER BY Failure_Rate_Pct DESC;


-- --------------------------------------------------------------------
-- Query 2: Thermal Stress Analysis (Process vs Air Temp Diff)
-- --------------------------------------------------------------------
SELECT 
    Machine_Failure,
    COUNT(*) AS Sample_Count,
    ROUND(AVG(Air_temperature_K), 2) AS Avg_Air_Temp_K,
    ROUND(AVG(Process_temperature_K), 2) AS Avg_Process_Temp_K,
    ROUND(AVG(Temp_Diff), 2) AS Avg_Temp_Diff_K,
    ROUND(MAX(Temp_Diff), 2) AS Max_Temp_Diff_K,
    ROUND(MIN(Temp_Diff), 2) AS Min_Temp_Diff_K
FROM cleaned_sensor_features
GROUP BY Machine_Failure;


-- --------------------------------------------------------------------
-- Query 3: Mechanical Power Distribution per Failure Mode
-- --------------------------------------------------------------------
SELECT 
    CASE 
        WHEN TWF = 1 THEN 'Tool Wear Failure'
        WHEN HDF = 1 THEN 'Heat Dissipation Failure'
        WHEN PWF = 1 THEN 'Power Failure'
        WHEN OSF = 1 THEN 'Overstrain Failure'
        WHEN RNF = 1 THEN 'Random Failure'
        ELSE 'No Failure'
    END AS Failure_Category,
    COUNT(*) AS Count,
    ROUND(AVG(Power_W), 2) AS Avg_Power_Watts,
    ROUND(AVG(Torque_Nm), 2) AS Avg_Torque_Nm,
    ROUND(AVG(Rotational_speed_rpm), 2) AS Avg_RPM
FROM cleaned_sensor_features
GROUP BY Failure_Category
ORDER BY Avg_Power_Watts DESC;


-- --------------------------------------------------------------------
-- Query 4: Top 10 Most Severely Worn Machines at Failure (Window Function: RANK)
-- --------------------------------------------------------------------
WITH RankedFailures AS (
    SELECT 
        UDI,
        Product_ID,
        Type,
        Tool_wear_min,
        Torque_Nm,
        Machine_Failure,
        RANK() OVER (PARTITION BY Type ORDER BY Tool_wear_min DESC) AS Rank_In_Type
    FROM cleaned_sensor_features
    WHERE Machine_Failure = 1
)
SELECT UDI, Product_ID, Type, Tool_wear_min, Torque_Nm, Rank_In_Type
FROM RankedFailures
WHERE Rank_In_Type <= 3
ORDER BY Type, Rank_In_Type;


-- --------------------------------------------------------------------
-- Query 5: Moving Average of Operating Parameters (Window Function: Moving Avg)
-- --------------------------------------------------------------------
SELECT 
    UDI,
    Product_ID,
    Rotational_speed_rpm,
    Torque_Nm,
    ROUND(AVG(Rotational_speed_rpm) OVER (
        ORDER BY UDI 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) AS Moving_Avg_RPM_5,
    ROUND(AVG(Torque_Nm) OVER (
        ORDER BY UDI 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) AS Moving_Avg_Torque_5
FROM cleaned_sensor_features
LIMIT 20;


-- --------------------------------------------------------------------
-- Query 6: Cumulative Tool Wear per Product Type (Window Function: SUM OVER)
-- --------------------------------------------------------------------
SELECT 
    UDI,
    Product_ID,
    Type,
    Tool_wear_min,
    SUM(Tool_wear_min) OVER (
        PARTITION BY Type 
        ORDER BY UDI 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS Cumulative_Type_Tool_Wear
FROM cleaned_sensor_features
LIMIT 20;


-- --------------------------------------------------------------------
-- Query 7: High-Risk Machine Identification CTE
-- --------------------------------------------------------------------
WITH HighRiskCriteria AS (
    SELECT 
        UDI,
        Product_ID,
        Type,
        Temp_Diff,
        Power_W,
        Tool_wear_min,
        Torque_Nm,
        Machine_Failure
    FROM cleaned_sensor_features
    WHERE Temp_Diff < 8.6 AND Rotational_speed_rpm < 1380 -- HDF condition proxy
       OR Tool_wear_min > 200                              -- High tool wear risk
       OR Power_W > 9000 OR Power_W < 3500                -- PWF condition proxy
)
SELECT 
    Type,
    COUNT(*) AS High_Risk_Flags,
    SUM(Machine_Failure) AS Actual_Failures,
    ROUND(CAST(SUM(Machine_Failure) AS FLOAT) * 100.0 / COUNT(*), 2) AS Detection_Accuracy_Pct
FROM HighRiskCriteria
GROUP BY Type;


-- --------------------------------------------------------------------
-- Query 8: Failure Rate Binned by Tool Wear Ranges
-- --------------------------------------------------------------------
SELECT 
    CASE 
        WHEN Tool_wear_min < 50 THEN '0-50 min (Low)'
        WHEN Tool_wear_min BETWEEN 50 AND 120 THEN '50-120 min (Moderate)'
        WHEN Tool_wear_min BETWEEN 121 AND 200 THEN '121-200 min (High)'
        ELSE '>200 min (Critical)'
    END AS Wear_Bin,
    COUNT(*) AS Total_Machines,
    SUM(Machine_Failure) AS Failures,
    ROUND(CAST(SUM(Machine_Failure) AS FLOAT) * 100.0 / COUNT(*), 2) AS Failure_Rate_Pct
FROM cleaned_sensor_features
GROUP BY Wear_Bin
ORDER BY Failure_Rate_Pct ASC;


-- --------------------------------------------------------------------
-- Query 9: Multi-Failure Co-occurrence Matrix
-- --------------------------------------------------------------------
SELECT 
    (TWF + HDF + PWF + OSF + RNF) AS Total_Failure_Modes_Active,
    COUNT(*) AS Machine_Count,
    SUM(Machine_Failure) AS Confirmed_Machine_Failures
FROM cleaned_sensor_features
GROUP BY Total_Failure_Modes_Active
ORDER BY Total_Failure_Modes_Active DESC;


-- --------------------------------------------------------------------
-- Query 10: Product Type Reliability Ranking (CTE + Window Function DENSE_RANK)
-- --------------------------------------------------------------------
WITH ReliabilityMetrics AS (
    SELECT 
        Type,
        COUNT(*) AS Total_Inspected,
        SUM(Machine_Failure) AS Failure_Count,
        ROUND(1.0 - (CAST(SUM(Machine_Failure) AS FLOAT) / COUNT(*)), 4) AS Reliability_Score
    FROM cleaned_sensor_features
    GROUP BY Type
)
SELECT 
    Type,
    Total_Inspected,
    Failure_Count,
    Reliability_Score,
    DENSE_RANK() OVER (ORDER BY Reliability_Score DESC) AS Reliability_Rank
FROM ReliabilityMetrics;


-- --------------------------------------------------------------------
-- Query 11: Machine Maintenance Log Summary (Recent Prediction Logs)
-- --------------------------------------------------------------------
SELECT 
    Prediction_ID,
    UDI,
    Predicted_Class,
    ROUND(Failure_Probability, 4) AS Risk_Score,
    Risk_Tier,
    Timestamp
FROM maintenance_logs
ORDER BY Timestamp DESC
LIMIT 10;
