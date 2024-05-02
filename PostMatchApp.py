import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import altair as alt

lg_lookup = pd.read_csv("https://raw.githubusercontent.com/griffisben/Post_Match_App/main/PostMatchLeagues.csv")
league_list = sorted(lg_lookup.League.tolist())

with st.sidebar:
    league = st.selectbox('What League Do You Want Reports For?', league_list)
    update_date = lg_lookup[lg_lookup.League==league].Update.values[0]
    
st.title(f"{league} Post-Match Reports")
st.subheader(f"Last Updated: {update_date}\n")
st.subheader('All data via Opta. Created by Ben Griffis (@BeGriffis on Twitter)')
st.subheader('Note: you may use these visuals in any of your work, but you MUST give me credit and note that the data is from Opta.')

df = pd.read_csv(f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/League_Files/{league.replace(' ','%20')}%20Full%20Match%20List.csv")
df['Match_Name'] = df['Match'] + ' ' + df['Date']

with st.sidebar:
    team_list = sorted(list(set(df.Home.unique().tolist() + df.Away.unique().tolist())))
    team = st.selectbox('Team', team_list)

    match_list = df[(df.Home == team) | (df.Away == team)].copy()
    match_choice = st.selectbox('Match', match_list.Match_Name.tolist())

match_string = match_choice.replace(' ','%20')
url = f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/Image_Files/{league.replace(' ','%20')}/{match_string}.png"
response = requests.get(url)
game_image = Image.open(io.BytesIO(response.content))

team_data = pd.read_csv(f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/Stat_Files/{league.replace(' ','%20')}.csv")
team_data = team_data[team_data.Team==team][['Match','Date','Possession','Field Tilt','Passes in Opposition Half','Passes into Box','xT','Shots','Shots per 1.0 xT','PPDA','High Recoveries','Crosses','Corners','Fouls']].reset_index(drop=True)
team_data['Shots per 1.0 xT'] = team_data['Shots per 1.0 xT'].astype(float)
team_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)

report_tab, data_tab, graph_tab = st.tabs(['Match Report', 'Data by Match - Table', 'Data by Match - Graph'])

report_tab.image(game_image)
data_tab.write(team_data)
with graph_tab:
    var = st.selectbox('Metric to Plot', ['Possession','Field Tilt','Passes in Opposition Half','Passes into Box','xT','Shots','Shots per 1 xT','PPDA','High Recoveries','Crosses','Corners','Fouls'])
    st.write(f'{team} {var} By Match')
    c = (
       alt.Chart(team_data)
       .mark_line()
       .encode(x="Date", y=var, tooltip=['Match','Date',var,'Possession'])
        .mark_config(color='#4c94f6')
    )
    st.altair_chart(c, use_container_width=True)
