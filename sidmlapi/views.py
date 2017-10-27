from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from .serializers import UserSerializer, GroupSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
import json
import math
from django.http import HttpResponse, HttpResponseRedirect
import os

#import data science libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import json

import csv
import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class post_recommendation(APIView):

    def get(self, request, *args, **kw):
        summary = "Please use post to post json with itemId to get recommended itemId back. Thanks."
        result = {"result": summary}
        response = Response(result, status=status.HTTP_200_OK)
        return response

    def post(self, request, *args, **kw):
        msg_body = json.loads(request.body)
        print("msg_body: {}".format(msg_body))
        print(type(msg_body))
        csv_file = csv.reader(open('input/recommendations.csv', "rt", encoding= "utf8"), delimiter=",")
        itemId_get = msg_body["itemId"]
        recommend_itemId = []
        recommend_lift = []
        for row in csv_file:
            for i in itemId_get:
                if i == row[1]:
                    recommend_itemId.append(row[2])
                    recommend_lift.append(row[11]) 
        result = {"result":"success", "recommend_itemId": recommend_itemId, "recommend_lift": recommend_lift}
        response = Response(result, status=status.HTTP_200_OK)
        return response



class ShowAllPosRecommender(APIView):
    # Refer to here na krub https://www.kaggle.com/datatheque/association-rules-mining-market-basket-analysis
    def get(self,request,*args,**kw):
        # Function that returns the size of an object in MB
        def size(obj):
            return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))
        # A. Load order data
        orders = pd.read_csv('input/order_products__prior.csv')
        print('orders -- dimensions: {0};   size: {1}'.format(orders.shape, size(orders)))
        #display(orders.head())
        # B. Convert order data into format expected by the association rules function
        # Convert from DataFrame to a Series, with order_id as index and item_id as value
        orders = orders.set_index('order_id')['product_id'].rename('item_id')
        #display(orders.head(10))
        type(orders)
        #C. Display summary statistics for order data
        print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'.format(orders.shape, size(orders), len(orders.index.unique()), len(orders.value_counts())))
        
        # Returns frequency counts for items and item pairs
        def freq(iterable):
            if type(iterable) == pd.core.series.Series:
                return iterable.value_counts().rename("freq")
            else: 
                return pd.Series(Counter(iterable)).rename("freq")

            
        # Returns number of unique orders
        def order_count(order_item):
            return len(set(order_item.index))


        # Returns generator that yields item pairs, one at a time
        def get_item_pairs(order_item):
            order_item = order_item.reset_index().as_matrix()
            for order_id, order_object in groupby(order_item, lambda x: x[0]):
                item_list = [item[1] for item in order_object]
                      
                for item_pair in combinations(item_list, 2):
                    yield item_pair
                    

        # Returns frequency and support associated with item
        def merge_item_stats(item_pairs, item_stats):
            return (item_pairs
                        .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                        .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


        # Returns name associated with item
        def merge_item_name(rules, item_name):
            columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
                       'confidenceAtoB','confidenceBtoA','lift']
            rules = (rules
                        .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                        .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
            return rules[columns]               


        def association_rules(order_item, min_support):

            print("Starting order_item: {:22d}".format(len(order_item)))


            # Calculate item frequency and support
            item_stats             = freq(order_item).to_frame("freq")
            item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


            # Filter from order_item items below min support 
            qualifying_items       = item_stats[item_stats['support'] >= min_support].index
            order_item             = order_item[order_item.isin(qualifying_items)]

            print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
            print("Remaining order_item: {:21d}".format(len(order_item)))


            # Filter from order_item orders with less than 2 items
            order_size             = freq(order_item.index)
            qualifying_orders      = order_size[order_size >= 2].index
            order_item             = order_item[order_item.index.isin(qualifying_orders)]

            print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
            print("Remaining order_item: {:21d}".format(len(order_item)))


            # Recalculate item frequency and support
            item_stats             = freq(order_item).to_frame("freq")
            item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


            # Get item pairs generator
            item_pair_gen          = get_item_pairs(order_item)


            # Calculate item pair frequency and support
            item_pairs              = freq(item_pair_gen).to_frame("freqAB")
            item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

            print("Item pairs: {:31d}".format(len(item_pairs)))


            # Filter from item_pairs those below min support
            item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

            print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


            # Create table of association rules and compute relevant metrics
            item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
            item_pairs = merge_item_stats(item_pairs, item_stats)
            
            item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
            item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
            item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
            
            
            # Return association rules sorted by lift in descending order
            return item_pairs.sort_values('lift', ascending=False)

        rules = association_rules(orders, 0.01)  
        # Replace item ID with item name and display association rules
        item_name   = pd.read_csv('input/products.csv')
        item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
        rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)
        # display(rules_final)

        summary = rules_final
        result = {"result": summary}
        response = Response(result, status=status.HTTP_200_OK)
        return response



