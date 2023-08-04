#!/usr/bin/env python
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example illustrates how to get campaign criteria.

Retrieves negative keywords in a campaign.
"""

import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException


def create_performance_report(customer_id):

    client = GoogleAdsClient.load_from_storage("./keyword-performance-report.yaml")

    ga_service = client.get_service("GoogleAdsService")

    query = """
        SELECT
            ad_group_criterion.keyword.text,
            ad_group_criterion.keyword.match_type,
            ad_group.name,
            ad_group.id,
            campaign.name,
            campaign.id,
            metrics.conversions,
            metrics.cost_micros,
            metrics.cost_per_conversion,
            metrics.clicks,
            metrics.impressions,
            metrics.ctr, 
            metrics.average_cpc,
            metrics.conversions_from_interactions_rate
        FROM keyword_view WHERE segments.date DURING LAST_7_DAYS
        AND campaign.advertising_channel_type = 'SEARCH'
        AND ad_group.status = 'ENABLED'
        AND ad_group_criterion.status IN ('ENABLED', 'PAUSED')
        ORDER BY metrics.impressions DESC
        LIMIT 100"""

    # Issues a search request using streaming.
    search_request = client.get_type("SearchGoogleAdsStreamRequest")
    search_request.customer_id = customer_id
    search_request.query = query
    stream = ga_service.search_stream(search_request)

    data = []
    for batch in stream:
        for row in batch.results:
            inner_data = []
            campaign = row.campaign
            ad_group = row.ad_group
            criterion = row.ad_group_criterion
            metrics = row.metrics

            # Keyword Name for e.g "game development"
            inner_data.append(criterion.keyword.text) 

            # Keyword Match Type
            match_type = criterion.keyword.match_type
            if match_type == 0:
                inner_data.append("Unspecified match")
            elif match_type == 1:
                inner_data.append("Unknown match")
            elif match_type == 2:
                inner_data.append("Exact match")
            elif match_type == 3:
                inner_data.append("Phase match")
            elif match_type == 4:
                inner_data.append("Broad match")

            # Ad Group Name of this keyword for e.g "Ad group 1"
            inner_data.append(ad_group.name) 

            # Ad Group Id of this keyword for e.g "152490121432"
            inner_data.append(str(ad_group.id))

            # Campaign Name of this keyword for e.g "Leads-Search-1"
            inner_data.append(campaign.name) 

            inner_data.append(str(campaign.id))

            # Conversions for this keyword
            inner_data.append(metrics.conversions)

            # Total Cost for this keyword
            inner_data.append(metrics.cost_micros)

            # Cost per conversion for this keyword
            if metrics.conversions == 0.0:
                inner_data.append(0)
            else:
                inner_data.append(metrics.cost_per_conversion)

            # Total Clicks for this keyword
            inner_data.append(metrics.clicks)

            # Total Impressions for this keyword
            inner_data.append(metrics.impressions)

            # CTR (clicks/impressions) percentage for this keyword
            if metrics.impressions == 0:
                inner_data.append("0%")
            else:
                inner_data.append(metrics.ctr)

            # Avg. CPC for this keyword
            if metrics.clicks == 0:
                inner_data.append(0.00)
            else:
                inner_data.append(metrics.average_cpc)

            # Avg. CPC for this keyword
            if metrics.clicks == 0:
                inner_data.append("0%")
            else:
                inner_data.append(metrics.conversions_from_interactions_rate)

            data.append(inner_data)

    df = pd.DataFrame(
        data,
        columns=[
            "Keyword", 
            "Match Type", 
            "Ad Group", 
            "Ad Group Id",
            "Campaign", 
            "Campaign Id",
            "Conversions",
            "Cost",
            "Cost / conversions",
            "Clicks", 
            "Impressions",
            "CTR",
            "Average CPC",
            "Conv. rate"
            ]
    )
    return df

def get_customer_number(customer_id):
    # Assuming that the customer ID format is "customers/{number}"
    # We can extract the number part and return it.
    return customer_id.split('/')[-1]

# Streamlit app
def generate_performance_report():
    st.title("Keyword Performance Report")

    client = GoogleAdsClient.load_from_storage("./keyword-performance-report.yaml")

    customer_service = client.get_service("CustomerService")

    customer_ids_list = (customer_service.list_accessible_customers(client.get_type("ListAccessibleCustomersRequest")).resource_names)

    # # Render the Customer Ids List
    # st.write("#### Accessible Customer Ids List:")
    # st.markdown("\n".join(f"- {get_customer_number(id)}" for id in customer_ids_list))

    st.write("#### Enter your Google Ads Customer ID:")

    #For e.g 7197615188

    customer_id = st.text_input("Customer ID", "7197615188")

    if st.button("Retrieve Data"):
        if customer_id:
            try:
                # Retrieve data using the Google Ads API and convert it into a DataFrame
                data = create_performance_report(customer_id)

                # Display the DataFrame as an Excel table
                st.write(data)
            except GoogleAdsException as ex:
                st.error(f"An error occurred: {ex}")
        else:
            st.warning("Please enter a valid Customer ID.")

# if __name__ == "__main__":
#     main()





      