SELECT Region, Sales, SUM(DISTINCT A), AVG(B), COUNT(ALL BAR) FROM BAZ GROUP BY Region, Sales;