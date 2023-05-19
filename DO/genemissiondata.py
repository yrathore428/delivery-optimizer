import csv
import random as rd
import pandas as pd
import numpy as np

MIN_DELIVERY_DAYS = 1
MAX_DELIVERY_DAYS = 21
MIN_REVIEW_SCORE = 0
MAX_REVIEW_SCORE = 9
MIN_RANK = 0
MAX_RANK = 4
MIN_FRAGILITY = 0
MAX_FRAGILITY = 2
STORAGE_TIME_CUTOFF = 90
MIN_TRAFFIC_LEVEL = 0
MAX_TRAFFIC_LEVEL = 9
DESTINATION_CATEGORY_LIST = ["city", "country", "continent", "world"]
ROAD_CONDITIONS = ["poor", "average", "good"]
LAND_VEHICLES = ["train", "truck", "van"]
FUEL = ["diesel", "petrol", "green_petrol"]
MIN_LONG_DISTANCE = 200
MAX_LONG_DISTANCE = 5000
MIN_SHORT_DISTANCE = 1
MAX_SHORT_DISTANCE = 500
PLANE_EFFICIENCY = 0.7
SHIP_EFFICIENCY = 0.8
TRAIN_EFFICIENCY = 0.9
TRUCK_EFFICIENCY = 0.88
VAN_EFFICIENCY = 0.93
PETROL_EMISSION_FACTOR = 0.4
GREENPETROL_EMISSION_FACTOR = 0.25
DIESEL_EMISSION_FACTOR = 0.6
MIN_COST = 1.3
MAX_COST = 9.6

def calculate_emissions(distance, vehicle, fuel):
            match vehicle:
    
             case "train":
                if(fuel == "petrol"):
                    emissions = (distance * PETROL_EMISSION_FACTOR)/TRAIN_EFFICIENCY
                elif(fuel == "diesel"):
                    emissions = (distance * DIESEL_EMISSION_FACTOR)/TRAIN_EFFICIENCY
                else:
                    emissions = (distance * GREENPETROL_EMISSION_FACTOR)/TRAIN_EFFICIENCY

             case "truck":
                if(fuel == "petrol"):
                    emissions = (distance * PETROL_EMISSION_FACTOR)/TRUCK_EFFICIENCY
                elif(fuel == "diesel"):
                    emissions = (distance * DIESEL_EMISSION_FACTOR)/TRUCK_EFFICIENCY
                else:
                    emissions = (distance * GREENPETROL_EMISSION_FACTOR)/TRUCK_EFFICIENCY
             
             case "van":
                if(fuel == "petrol"):
                    emissions = (distance * PETROL_EMISSION_FACTOR)/VAN_EFFICIENCY
                elif(fuel == "diesel"):
                    emissions = (distance * DIESEL_EMISSION_FACTOR)/VAN_EFFICIENCY
                else:
                    emissions = (distance * GREENPETROL_EMISSION_FACTOR)/VAN_EFFICIENCY
             
             case "plane":
                if(fuel == "petrol"):
                    emissions = (distance * PETROL_EMISSION_FACTOR)/PLANE_EFFICIENCY
                elif(fuel == "diesel"):
                    emissions = (distance * DIESEL_EMISSION_FACTOR)/PLANE_EFFICIENCY
                else:
                    emissions = (distance * GREENPETROL_EMISSION_FACTOR)/PLANE_EFFICIENCY
             
             case "ship":
                if(fuel == "petrol"):
                    emissions = (distance * PETROL_EMISSION_FACTOR)/SHIP_EFFICIENCY
                elif(fuel == "diesel"):
                    emissions = (distance * DIESEL_EMISSION_FACTOR)/SHIP_EFFICIENCY
                else:
                    emissions = (distance * GREENPETROL_EMISSION_FACTOR)/SHIP_EFFICIENCY

            return emissions

# Column headers
headers = ["destination_category", "mode", "customer_review", "carrier_rank", 
           "fragility", "atseal", "traffic_level", "road_condition", "deadline", 
           "distance", "transport_vehicle", "fuel_type", "emissions", "dcost",
           "delivery_ontime", "dtime"]

# Open CSV file for writing
with open("emission_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write column headers
    writer.writerow(headers)

    for i in range(1000):
        #assigning minimum cost
        dcost = MIN_COST
        
        destination_category = rd.choice(DESTINATION_CATEGORY_LIST)
        
        #deadline (based on destination category, change from just being random)
        match destination_category:
            case "city":
                deadline = rd.randint(1,2)
                distance = rd.randint(MIN_SHORT_DISTANCE, MAX_LONG_DISTANCE)
            case "country":
                deadline = rd.randint(3,7)
                distance = rd.randint(MIN_LONG_DISTANCE, MAX_SHORT_DISTANCE)
                dcost += MIN_COST
            case "continent":
                deadline = rd.randint(8,13)
                distance = rd.randint(MIN_LONG_DISTANCE, MAX_LONG_DISTANCE)
                dcost += MIN_COST*2
            case "world":
                deadline = rd.randint(14,21)
                distance = rd.randint(MAX_SHORT_DISTANCE, MAX_LONG_DISTANCE)
                dcost += MIN_COST*3
        
        
        #fragility
        fragility = rd.randint(MIN_FRAGILITY, MAX_FRAGILITY)
        #delivery mode
        if(deadline < 8 or fragility > 1):
            delivery_mode = 'air'
            transport_vehicle = "plane"
            fuel_type = "petrol"
            emissions = calculate_emissions(distance, transport_vehicle, fuel_type)
            dcost += MIN_COST/2
        elif(deadline > 7 and deadline < 15):
            delivery_mode = 'land'
            transport_vehicle = rd.choice(LAND_VEHICLES)
            fuel_type = rd.choice(FUEL)
            emissions = calculate_emissions(distance, transport_vehicle, fuel_type)
        else:
            delivery_mode = 'sea'
            transport_vehicle = "ship"
            fuel_type = rd.choice(FUEL)
            emissions = calculate_emissions(distance, transport_vehicle, fuel_type)

        traffic_level = rd.randint(MIN_TRAFFIC_LEVEL, MAX_TRAFFIC_LEVEL)

        #delivery time (also add the effect of traffic)
        if(delivery_mode == 'sea'):
            dtime = rd.randint(15, 24)
        elif(delivery_mode == 'land'):
            dtime = rd.randint(8, 17)
            if(traffic_level < 4):
                dtime -= 1
            elif(traffic_level >= 4 and traffic_level < 7):
                dtime += 1
                dcost += MIN_COST/2
            else:
                dtime += 2
                dcost += MIN_COST
        else:
            dtime = rd.randint(1, 9)

        
        atseal = rd.randint(0, 1)
        if(atseal == 1):
            dcost += MIN_COST

        # effect of road conditions
        road_condition = rd.choice(ROAD_CONDITIONS)
        match road_condition:
            case "poor":
                dtime += 1
                dcost += MIN_COST
            case "good":
                dtime -= 1
        
        
        # carrier rank and its effect 
        carrier_rank = rd.randint(MIN_RANK, MAX_RANK)
        if(carrier_rank == 2):
            dtime -= 1
            dcost += MIN_COST
        elif(carrier_rank > 2):
            dtime -= 2
            dcost += MIN_COST*2
        elif(carrier_rank == 0):
            dtime += 1
        
        if(dtime <= 0):
            dtime = 1

        # delivery_ontime
        if(dtime <= deadline):
            delivery_ontime = 1
        else:
            delivery_ontime = 0
        
        if(dcost > MAX_COST):
            dcost = MAX_COST

        #customer_review
        if(delivery_ontime == 1):
            customer_review = rd.randint(5, 9)
        else:
            customer_review = rd.randint(0,4)


        writer.writerow([destination_category, delivery_mode, customer_review, carrier_rank, 
                         fragility, atseal, traffic_level, road_condition, deadline, distance, 
                         transport_vehicle, fuel_type, emissions, dcost,
                         delivery_ontime, dtime])



# destination location and/or location category (1 to 4: within city, within country, within continent, outside continent)
# road conditions: similar to traffic level
# carrier data/rank: can replace product rank. this can replace product rank. good rank means
# good ontime delivery
# delivery cost
# supply chain data (inventory status, shipping schedule, and production schedule)
# for example depending on inventory status, shiping can be delayed if product is out of stock or 
# there needs to be a wait time before the delivery can be dispatched

