import os
import sys


import json
import pandas as pd
import numpy as np

#load environment variables
import ast
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
G_KEY = os.environ["G_KEY"]
G_SCOPES = ast.literal_eval(os.environ["G_SCOPES"])
G_PRJ_ID = os.environ["G_PRJ_ID"]

G_SCOPES = ["https://www.googleapis.com/auth/bigquery","https://www.googleapis.com/auth/drive.readonly",]
type(G_SCOPES)


from google.cloud import bigquery
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(G_KEY, scopes=G_SCOPES)
client = bigquery.Client(credentials= credentials,project=G_PRJ_ID)

try:
    sql = """SELECT * FROM `quant-349811.alt.asx_news_pdftxt` pdf left outer join `alt.asx_news` a on a.News_URL = pdf.url where a.Symbol = 'CHN.AU' and extract(YEAR from a.Date) in (2020,2021)"""
    df_Load = pd.read_gbq(sql, project_id=G_PRJ_ID,credentials=credentials)
    
except:
    print("Error fetching data from BigQuery")

df_Load['clean_text'] = ''
df_Load['summary'] = ''
df_Load['topic_id'] = ''
df_Load['topic_text'] = ''

#a function that takes a json, and adds a new column to the dataframe if the column name does not exists
def add_columns_from_structure(df, structure):
    for key, value in structure.items():
        if key not in df.columns:
            if isinstance(value, list):
                df[key] = np.nan
            else:
                df[key] = value

data_collection = {
    "datatype": "",
    "specific_datatype": [],
    "stage": "",
    "analysis_method": "",
    "prospect_name": "",
    "main_commodity": "",
    "deposit_style": ""
}

new_lisence = {
    "name": "",
    "datatypes": [],
    "lisence_type": "",
    "seller_name": "",
    "deposit_style": "",
    "main_commodity": [],
    "location": "",
    "area": ""
}



add_columns_from_structure(df_Load, data_collection)
add_columns_from_structure(df_Load, new_lisence)

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
chat4 = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chat3 = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#4096 tokens = 16384, BUT crashes on 12000 characters. That is 0.7 of the total tokens
chat3_long = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k")
#16384 tokens = 65536 char

#pulling first row for testing
#for index, row in df_Load[0:1].iterrows():
for index, row in df_Load.iterrows():
    url = row['url']
    headline = row['headline']
    raw_text = row['text']

    if index < 4:
        continue

    #trim the length of the input string if above 32k characters. Most of the important stuff is found in the beginning
    #if len(raw_text) > 12000:
    #    raw_text = raw_text[0:int(65536*0.7)]


    #len(raw_text[0:int(16384*4*0.7)])


    #clean the text
    result = chat3_long(
        [
            SystemMessage(content="You are a stock analyst with expertise in mineral exploration. The text in the press release is messy. I want you to clean it up so it is easy to read Ignore disclaimers and forward looking statements. Here is the document: "),
            HumanMessage(content=raw_text[0:int(16384*2)]) # 1 token is 4 characters, but we multiply by 2.2 because we also need to count the output
        ]
    )

    clean_text = result.content
    print(clean_text)
    #len(clean_text)
    row['clean_text'] = clean_text

    #summarize the text
    result = chat3_long(
        [
            SystemMessage(content="You are a stock analyst with expertise in mineral exploration. Get me the highlights from this press release. A short summary followed by up to 15 bulletpoints with important facts. Ignore disclaimers and forward looking statements. Here is the document: "),
            HumanMessage(content=clean_text) # 1 token is 4 characters, but we multiply by 3 because we also need to count the output
        ]
        
    )

    summary = result.content
    print(summary)

    row['summary'] = summary

    #get the topic 
    system_msg = """You are an analyst and I want you to find the main topic of the press releases. Focus on the information in the headline or in the highlights or summary of the presentation. Ignore disclaimers on forward looking statements.

        Topic:
        The topics of the press release usually fall into ONE of the following categories. You must find the most likely main topic: 
        (notation: topic_id. topic_text: explainer)
        1. Corporate Presentation: This category includes company presentations, corporate presentation, investor presentations, capital markets day or some kind of presentation given at an event or conference.
        2. Data collection: Examples are drilling results, start of drilling, field work, mapping, sampling of rocks, soils or collecting geophysical data. 
        3. New license: Is the company announcing a new lisence in their portfolio? (terms as property, tenement often used instead of lisence)
        4. Disposed lisence: Has the company sold or disposed of a lisence?
        5. Other: If none of the above does not fit, then return: Other: [The topic you think best describes]

        return your answers only as json on the form below:

        {
        "topic_id": "",
        "topic_text": ""
        }
        """


    result = chat3(
        [
            SystemMessage(content=system_msg),
            HumanMessage(content=summary)
        ]
    )

    topic = result.content
    print(topic)
    row['topic_id'] = json.loads(topic)['topic_id']
    row['topic_text'] = json.loads(topic)['topic_text']

    #get more info if category 

    topic_id = json.loads(topic)['topic_id']

    if topic_id in ["2"]:
        system_msg =  """
    
        You are an analyst of press releases from mineral exploration companies. 
        
        I will give you 4 tasks:

        1) I want you to read the press release and determine what datatype is the main topic.

        I want you to classify the datatype collected into one of the following categories
        a) Geochemical sampling: Examples are Grab Sampling,Soil Samples,Stream Sediment,Trench Sampling,Channel Sampling,Rock Chip sampling. The purpose is to measure the chemical composition of rocks and sediments.
        b) Feld work: Examples are Reconnaissance and Geological Mapping. It includes putting geologists on the ground to better understand the geology in the lisence. 
        c) Geophysical data: Examples are Magnetics,EM (Electromagnetics),Passive Seismic,MT (Magnetotellurics),IP (Induced Polarization),Conductivity,Radiometry,Ground Gravity Surveys,Airborne Gravity,Seismic ,Hyperspectral Imaging
        d) Drilling: Examples are Drilling, Drill Core,Diamond Drilling ,RC Drilling ,Aircore Drilling. The purpose is to measure the extent, volume and grade of a potential mineral deposit.
        

        2) I want you to classify the specific datatypes mentioned above.
        

        3) I want you to classify the stage of the datacollection
        More info on the stages
        a) Start: The press release announces that data collection will start in near future. Logically no results are presented if the stage is Start!
        b) Results: The press release announces actual resutls of one of the datatypes collected.

        4) If the datatype is geochemichal or drilling, i want you to find the analysis method. The two options are Assays or XRF and it is Assay unless XRF is mentioned.

        5) The name of the prospect, project or lisence. Shorten the name so that Flower Lake Lithium Project => Flower Lake

        6) The main commodity they are exploring for on the prospect. Return chemical element on this format Ni, Co, Cu, Au, REE, Li

        7) What is the deposit style, if known: Examples are 
        a) NiCuPGE: also called Nickel-copper-sulphide or PGE
        b) SEDEX: also called Sedimentary exhalative deposits
        c) VMS: Also called Volcanogenic massive sulfide
        d) Porphery
        e) IOCG: also called Iron oxide copper gold
        f) REE: Also called rare earth or any of the elements La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Sc, Y
        g) Litium Pegmatite

        return your answers only as json on the form below:

        {
        "datatype": "",
        "specific_datatype": [],
        "stage": "",
        "analysis_method": "",
        "prospect_name": "",
        "main_commodity": "",
        "deposit_style": ""
        }
        """
        result = chat4(
            [
                SystemMessage(content=system_msg),
                HumanMessage(content=summary)
            ]
        )

        topic2 = result.content
        print(topic2)
        try:
            # Loading the JSON data
            data_collection_data = json.loads(topic2)
            
            # Mapping values for data_collection
            row['datatype'] = data_collection_data.get('datatype', np.nan)
            row['specific_datatype'] = data_collection_data.get('specific_datatype', np.nan)
            row['stage'] = data_collection_data.get('stage', np.nan)
            row['analysis_method'] = data_collection_data.get('analysis_method', np.nan)
            row['prospect_name'] = data_collection_data.get('prospect_name', np.nan)
            row['main_commodity'] = data_collection_data.get('main_commodity', np.nan)
            row['deposit_style'] = data_collection_data.get('deposit_style', np.nan)

        except json.JSONDecodeError:
            print(f"Failed to decode JSON for data_collection at row {index}")


    elif topic_id in ["3"]:
        system_msg =  """
    
        You are an analyst of press releases from mineral exploration companies. I want you to read the press release and get more information about the new lisence the company announced
        
        I will give you 4 tasks:


        1) Name: (name of the lisence, tenement or property=
        2) Datatypes: (datatypes that are already collected in the lisence area Examples are Magnetics,EM (Electromagnetics),Passive Seismic,MT (Magnetotellurics),IP (Induced Polarization),Conductivity,Radiometry,Ground Gravity Surveys,Airborne Gravity,Seismic ,Hyperspectral Imaging, drilling, grab samples, chip samples, soil or sediment samples or drilling)
        3) Lisence type: (was the lisence based on a new application (new application) or is the lisence acquired from another company (acquired)) 
        4) Name of seller (company that sold the lisence, if the lisence was acquired)
        5) What is the prosepctive deposit style, if known: Examples are 
            a) NiCuPGE: also called Nickel-copper-sulphide or PGE
            b) SEDEX: also called Sedimentary exhalative deposits
            c) VMS: Also called Volcanogenic massive sulfide
            d) Porphery
            e) IOCG: also called Iron oxide copper gold
            f) REE: Also called rare earth or any of the elements La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Sc, Y
            g) Litium Pegmatite
        


        6) Name of main commodity (examples: Li, Cu, REE, Co, Zn)
        7) Location (describe where the lisence is located)
        8) Area (Name of the state or broader region)
        

        return your answers only as json on the form below:

        {
        "name": "",
        "datatypes": [],
        "lisence_type": "",
        "seller_name": "",
        "deposit_style": "",
        "main_commodity": [],
        "location": "",
        "area": ""
        }
        """
        result = chat4(
            [
                SystemMessage(content=system_msg),
                HumanMessage(content=summary)
            ]
        )

        topic2 = result.content
        print(topic2)
        try:
            # Loading the JSON data
            new_lisence_data = json.loads(topic2)
            
            # Mapping values for new_lisence
            row['name'] = new_lisence_data.get('name', np.nan)
            row['datatypes'] = new_lisence_data.get('datatypes', np.nan)
            row['lisence_type'] = new_lisence_data.get('lisence_type', np.nan)
            row['seller_name'] = new_lisence_data.get('seller_name', np.nan)
            row['deposit_style'] = new_lisence_data.get('deposit_style', np.nan)
            row['main_commodity'] = new_lisence_data.get('main_commodity', np.nan)
            row['location'] = new_lisence_data.get('location', np.nan)
            row['area'] = new_lisence_data.get('area', np.nan)

        except json.JSONDecodeError:
            print(f"Failed to decode JSON for new_lisence at row {index}")

        


        