

import time
import math
import openai
import pybase64
import pandas as pd
import streamlit as st
from decimal import Decimal
from googlesearch import search
from prompts.keywordPlanner import keywordPlanner
from google.ads.googleads.client import GoogleAdsClient
from prompts.keywordPlannerInput import keywordPlannerInput
from keyword_performance_report import generate_performance_report

decimal_bid_conversion = 6
wait_time_to_hit_google = 5

# [START generate_keyword_ideas]
def get_keyword_data(
    client, customer_id, location_ids, language_id, keyword_texts, page_url
):
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    keyword_competition_level_enum = client.get_type(
        "KeywordPlanCompetitionLevelEnum"
    ).KeywordPlanCompetitionLevel
    keyword_plan_network = client.get_type(
        "KeywordPlanNetworkEnum"
    ).KeywordPlanNetwork.GOOGLE_SEARCH_AND_PARTNERS
    location_rns = _map_locations_ids_to_resource_names(client, location_ids)
    language_rn = client.get_service("GoogleAdsService").language_constant_path(
        language_id
    )
    
    # keyword_annotation = client.enums.KeywordPlanKeywordAnnotationEnum
    
    # Either keywords or a page_url are required to generate keyword ideas
    # so this raises an error if neither are provided.
    if not (keyword_texts or page_url):
        raise ValueError(
            "At least one of keywords or page URL is required, "
            "but neither was specified."
        )
    
    
    
    # Only one of the fields "url_seed", "keyword_seed", or
    # "keyword_and_url_seed" can be set on the request, depending on whether
    # keywords, a page_url or both were passed to this function.
    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id
    request.language = language_rn
    request.geo_target_constants.extend(location_rns)
    request.include_adult_keywords = False
    request.keyword_plan_network = keyword_plan_network
    # request.keyword_annotation.extend(keyword_annotation)
    
    
    
    # To generate keyword ideas with only a page_url and no keywords we need
    # to initialize a UrlSeed object with the page_url as the "url" field.
    if not keyword_texts and page_url:
        request.url_seed.url = page_url
 
    # To generate keyword ideas with only a list of keywords and no page_url
    # we need to initialize a KeywordSeed object and set the "keywords" field
    # to be a list of StringValue objects.
    if keyword_texts and not page_url:
        request.keyword_seed.keywords.extend(keyword_texts)
 
    # To generate keyword ideas using both a list of keywords and a page_url we
    # need to initialize a KeywordAndUrlSeed object, setting both the "url" and
    # "keywords" fields.
    if keyword_texts and page_url:
        request.keyword_and_url_seed.url = page_url
        request.keyword_and_url_seed.keywords.extend(keyword_texts)
 
    keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(
        request=request
    )
    
    list_keywords = []
    for idea in keyword_ideas:
        # competition_value = idea.keyword_idea_metrics.competition.name
        list_keywords.append(idea)
    
    return list_keywords
 
def map_keywords_to_string_values(client, keyword_texts):
    keyword_protos = []
    for keyword in keyword_texts:
        string_val = client.get_type("StringValue")
        string_val.value = keyword
        keyword_protos.append(string_val)
    return keyword_protos
 
def _map_locations_ids_to_resource_names(client, location_ids):
    """Converts a list of location IDs to resource names.
    Args:
        client: an initialized GoogleAdsClient instance.
        location_ids: a list of location ID strings.
    Returns:
        a list of resource name strings using the given location IDs.
    """
    build_resource_name = client.get_service(
        "GeoTargetConstantService"
    ).geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]

def generate_keyword_ideas(client,customer_id,location_ids, language_id, ad_groups):


    web_links = []
    list_keywords = []

    for i in ad_groups:
        search_results = search(i, num_results=1)
        for j in search_results:
            web_links.append(j)

    for link in web_links:
        content = get_keyword_data(
            client,
            customer_id,
            location_ids,
            language_id,
            [],
            link
        )
        time.sleep(wait_time_to_hit_google)
        for x in range(len(content)):
            list_keywords.append(content[x])

    list_to_excel = []
    for x in range(len(list_keywords)):
        list_months = []
        list_searches = []
        list_annotations = []
        months = [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"
        ]
        for y in list_keywords[x].keyword_idea_metrics.monthly_search_volumes:
            month_num = y.month - 1 if (y.month - 1) < 12 else 0
            month_year = str(y.year) if (y.month - 1) < 12 else str(y.year+1)
            list_months.append(str(months[month_num]) + " - " + month_year)
            list_searches.append(y.monthly_searches)

        for y in list_keywords[x].keyword_annotations.concepts:
            list_annotations.append(y.concept_group.name)

        competition_index = int(list_keywords[x].keyword_idea_metrics.competition_index)
        competition_value = ""

        if competition_index >= 0 and competition_index <= 33:
            competition_value = "LOW"
        elif competition_index >= 34 and competition_index <= 66:
            competition_value = "MEDIUM"
        elif competition_index >= 67 and competition_index <= 100:
            competition_value = "HIGH"

        low_bid = (
            Decimal(list_keywords[x].keyword_idea_metrics.low_top_of_page_bid_micros)
            / Decimal(math.pow(10, decimal_bid_conversion))
        )
        high_bid = (
            Decimal(list_keywords[x].keyword_idea_metrics.high_top_of_page_bid_micros)
            / Decimal(math.pow(10, decimal_bid_conversion))
        )

        
        list_to_excel.append([
            (list_keywords[x].text), (list_keywords[x].keyword_idea_metrics.avg_monthly_searches),
            (competition_value), (list_keywords[x].keyword_idea_metrics.competition_index),
            (low_bid), (high_bid), (list_searches), (list_months), (list_annotations)
        ])

    return list_to_excel

def get_table_download_link(df, filename='download', file_format='csv'):
    """
    Generate a download link for a Pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be downloaded.
        filename (str): The name of the downloaded file (without extension).
                        Default is 'download'.
        file_format (str): The file format for download. Supported formats are 'csv', 'xlsx'.
                           Default is 'csv'.

    Returns:
        str: A string representing an HTML anchor tag with a link to download the file.
    """
    valid_formats = ['csv', 'xlsx']
    if file_format not in valid_formats:
        raise ValueError(f"Invalid file format. Supported formats are: {', '.join(valid_formats)}")

    if file_format == 'csv':
        data = df.to_csv(index=False)
        extension = 'csv'
    elif file_format == 'xlsx':
        data = df.to_excel(index=False)
        extension = 'xlsx'

    b64 = pybase64.b64encode(data.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/{file_format};base64,{b64}" download="{filename}.{extension}">**Download {extension.upper()} file**</a>'
    return href

def prioritize_keywords(keywords_data,average_searches:int):
    def get_past_searches(keyword):
        searches_past_months = eval(keyword["Searches Past Months"])
        return sum(searches_past_months) / len(searches_past_months) if searches_past_months else 0
    
    unique_keywords = set()
    sorted_keywords = []
    # Filter data where Average Searches are less than 100 and remove duplicates
    for keyword in keywords_data:
        if keyword["Average Searches"] >= average_searches and keyword["Keyword"] not in unique_keywords:
            unique_keywords.add(keyword["Keyword"])
            sorted_keywords.append(keyword)
    # Sort by low bid (ascending order)
    sorted_keywords = sorted(sorted_keywords, key=lambda x: x["Low Bid"])
    # Evaluate average searches (descending order)
    sorted_keywords = sorted(sorted_keywords, key=lambda x: x["Average Searches"], reverse=True)
    # Consider competition level (LOW > MEDIUM > HIGH)
    sorted_keywords = sorted(sorted_keywords, key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x["Competition Level"]))
    # Check past months' searches (consider highest average search volume)
    sorted_keywords = sorted(sorted_keywords, key=get_past_searches, reverse=True)
    # Assess competition index (ascending order)
    sorted_keywords = sorted(sorted_keywords, key=lambda x: x["Competition Index"])
    # Consider bid range (choose keywords within your budget range)
    # sorted_keywords = [kw for kw in sorted_keywords if kw["Low Bid"] <= max_bid]
    # Consider brand/non-brand annotations (prioritize brand keywords)
    # sorted_keywords = sorted(sorted_keywords, key=lambda x: "Brands" in x["List Annotations"], reverse=True)
    # Evaluate relevance (add your relevance criteria here if available)
    return sorted_keywords

def main():

    # Step 1: Initial UI - Choose between AI-generated ad word ideas or user-provided ad groups
    st.title("Keyword Analysis")
    option = st.radio("#### Select an option:", ("Generate ad group ideas using AI", "Use your own ad group ideas", "Generate Keyword Report"))

    user_api_key = ""

    # Initialize session state variables
    if 'ad_group_ai_result' not in st.session_state:
        st.session_state.ad_group_ai_result = ""

    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = False

    if option == "Generate ad group ideas using AI":
        # user_api_key
        user_api_key = st.text_input(
        label="#### Your OpenAI API key 👇",
        placeholder="Paste your openAI API key, sk-",
        type="password")
        
        ai_prompt = st.text_area("#### Enter AI prompt to generate ad group ideas:")
        if st.button("Submit AI Prompt"):
            # Call the AI API with the provided prompt and display the result
            if ai_prompt.strip() != "":
                openai.api_key = user_api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {
                        "role": "user",
                        "content": ai_prompt
                        },
                    ],
                    temperature=0.5,
                    
                    )
                ai_result = response.choices[0].message.content  # call the AI API
                st.session_state.ad_group_ai_result = ai_result
                st.session_state.show_sidebar = True
    
    elif option == "Use your own ad group ideas":
        st.session_state.show_sidebar = True
    elif option == "Generate Keyword Report":
        generate_performance_report()


    if st.session_state.show_sidebar:

        if st.session_state.ad_group_ai_result != "":
            st.write(st.session_state.ad_group_ai_result)

        # Customer Id
        default_customer_id = "8799968521"

        # Average Searches
        default_average_search = 100

        # Average Searches
        default_location_ids = []  # location ID for New York, NY (1023191)

        # Average Searches
        default_language_id = "" # language ID for English 1000 (English)

        # Number of ad Groups
        num_ad_groups = 0

        # List of Ad Groups Given By User
        ad_groups = []

        # Generated Keyword Data Sheet
        list_to_excel = []

        # Client Instance
        client = GoogleAdsClient.load_from_storage("./keyword-planner.yaml")


        # Language Selection Logic
        language_excel_sheet = pd.read_csv("./utilities/language.csv")
        language_excel_dict = language_excel_sheet.to_dict(orient="records")
        language_names = tuple(item['Language name'] + " (" + (item['Language code']) + ")"  for item in language_excel_dict)

        selected_language = st.sidebar.selectbox(
        key="lang_id",
        label='#### Language 👇',
        help="#### Refer Here -> https://developers.google.com/google-ads/api/data/codes-formats#languages",
        options=language_names,
        index=10)

        default_language_id = language_excel_dict[language_names.index(selected_language)]['Criterion ID']

        # Location Selection Logic
        location_excel_sheet = pd.read_csv("./utilities/location.csv")
        location_excel_dict = location_excel_sheet.to_dict(orient="records")
        location_names = tuple(item['Canonical Name'] + " (" + item['Target Type'] + ")" for item in location_excel_dict)

        # Take number of locations 
        st.sidebar.write("#### Enter the number of locations 👇")
        num_of_location_ids = st.sidebar.number_input("Enter number of location", min_value=0,max_value=5, step=1, value=0)

        # Take num_of_location_ids number of locations as input
        for i in range(num_of_location_ids):
            selected_location = st.sidebar.selectbox(
            key=f"location_id {i+1}",
            label=f"Location {i+1}",
            help="#### Refer Here -> https://developers.google.com/google-ads/api/reference/data/geotargets",
            options=location_names,
            index=20867)
            default_location_ids.append(location_excel_dict[location_names.index(selected_location)]['Criteria ID'])


        # Input number of Ad Groups
        if len(default_location_ids) != 0:
            st.write("#### Enter the number of ad groups:")
            num_ad_groups = st.number_input("Number of Ad Groups", min_value=0, step=1, value=0)
        
        # Enter num_ad_groups number of ad group
        for i in range(num_ad_groups):
            ad_group = st.text_input(f"Ad Group {i+1}")
            if len(ad_group) != 0:
                ad_groups.append(ad_group)

        # Generate Keyword file in list format
        if len(ad_groups) != 0 and st.button("Generate Keyword Ideas"):
            list_to_excel = generate_keyword_ideas(client,default_customer_id,default_location_ids,default_language_id,ad_groups)

        # Generate excel sheet dataframe
        if len(list_to_excel) > 0:
            df = pd.DataFrame(
                list_to_excel,
                columns=["Keyword", "Average Searches", "Competition Level", "Competition Index",
                        "Low Bid", "High Bid", "Searches Past Months", "Past Months", "List Annotations"]
            )
            st.dataframe(df)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        csv_file = st.sidebar.file_uploader("#### Upload CSV file and Ask Query", type="csv")

        if csv_file is not None:
                
                # filter by average searches
                default_average_search = st.number_input(
                key="avg_search",
                label="#### Filter by Average Searches 👇",
                value=default_average_search)
                
                # make excel sheet data
                excel_sheet = pd.read_csv(csv_file)
                excel_sheet = excel_sheet.to_dict(orient="records")
                priority_keyword_data = prioritize_keywords(excel_sheet,default_average_search)
                excel_sheet_dataframe = pd.DataFrame(priority_keyword_data)
                st.dataframe(excel_sheet_dataframe)

                st.write("#### Search Comparison on basis of Competition Index 👇")
                chart = {
                    "mark": "point",
                    "encoding": {
                        "x": {
                            "field": "Competition Index",
                            "type": "quantitative",
                        },
                        "y": {
                            "field": "Average Searches",
                            "type": "quantitative",
                        },
                        "color": {"field": "Competition Level", "type": "nominal"},
                        "shape": {"field": "Competition Level", "type": "nominal"},
                        "tooltip": [
                            {"field": "Keyword", "type": "nominal", "title":"Keyword"},
                            {"field": "Competition Index", "type": "quantitative"},
                            {"field": "Average Searches", "type": "quantitative"},
                            {"field": "Low Bid", "type": "quantitative"},
                            {"field": "High Bid", "type": "quantitative"},
                            # Add more tooltip fields if needed
                        ]
                    },
                }

                st.vega_lite_chart(
                    excel_sheet_dataframe, chart, theme="streamlit", use_container_width=True
                )

                # user_api_key
                if user_api_key == "":
                    user_api_key = st.text_input(
                    label="#### Your OpenAI API key 👇",
                    placeholder="Paste your openAI API key, sk-",
                    type="password")

                if user_api_key != "":

                    # Prompt
                    keywordPlanningPromptTemplate = keywordPlanner(1)

                    keywordPlanningPromptTemplate = st.text_area(label="#### Modify Keyword Planning Prompt",value=keywordPlanningPromptTemplate,height=400)

                    excel_sheet_dataframe_list = [excel_sheet_dataframe.columns.values.tolist()] + excel_sheet_dataframe.values.tolist()

                    keywordPlannerInputTemplate = keywordPlannerInput(excel_sheet_dataframe_list,ad_groups)

                    if st.button("Submit"):
                        if (ad_groups is not None) and (ad_groups != "" or len(ad_groups) > 0):
                            with st.spinner(text="In progress..."):
                                openai.api_key = user_api_key
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo-16k",
                                    messages=[
                                        {
                                        "role": "system",
                                        "content": keywordPlanningPromptTemplate
                                        },
                                        {
                                        "role": "user",
                                        "content": keywordPlannerInputTemplate
                                        },
                                    ],
                                    temperature=0.5,
                                    
                                    )
                                st.write(response.choices[0].message.content)



# Call the main function when the script is executed
if __name__ == "__main__":
    main()
