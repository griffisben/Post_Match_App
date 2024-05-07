import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import altair as alt

#########################
def ben_theme():
    return {
        'config': {
            'background': '#fbf9f4',
            # 'text': '#4a2e19',
            'mark': {
                'color': '#4c94f6',
            },
            'axis': {
                'titleColor': '#4a2e19',
                'labelColor': '#4a2e19',
            },
            'text': {
                'fill': '#4a2e19'
            },
            'title': {
                'color': '#4a2e19',
                'subtitleColor': '#4a2e19'
            }
        }
    }

# register the custom theme under a chosen name
alt.themes.register('ben_theme', ben_theme)

# enable the newly registered theme
alt.themes.enable('ben_theme')
################################

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
league_data = team_data.copy().reset_index(drop=True)
team_data = team_data[team_data.Team==team].reset_index(drop=True)
team_data['Shots per 1.0 xT'] = team_data['Shots per 1.0 xT'].astype(float)
team_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)

league_data['Shots per 1.0 xT'] = league_data['Shots per 1.0 xT'].astype(float)
league_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)


team_data['xG per 1 xT'] = team_data['xG']/team_data['xT']
league_data['xG per 1 xT'] = league_data['xG']/league_data['xT']

team_data['xGA per 1 xT Against'] = team_data['xGA']/team_data['xT Against']
league_data['xGA per 1 xT Against'] = league_data['xGA']/team_data['xT Against']

if league in ['Saudi Pro League','Eredivisie','Irish Premier Division']:
    team_data['xG per 1 xT'] = team_data['xG']/team_data['xT']
    team_data['xGA per 1 xT Against'] = team_data['xGA']/team_data['xT Against']
    team_data['Result'] = 'D'
    team_data['Result'] = ['W' if team_data['Goals'][i]>team_data['Goals Conceded'][i] else team_data['Result'][i] for i in range(len(team_data))]
    team_data['Result'] = ['L' if team_data['Goals'][i]<team_data['Goals Conceded'][i] else team_data['Result'][i] for i in range(len(team_data))]
    league_data['Result'] = 'D'
    league_data['Result'] = ['W' if league_data['Goals'][i]>league_data['Goals Conceded'][i] else league_data['Result'][i] for i in range(len(league_data))]
    league_data['Result'] = ['L' if league_data['Goals'][i]<league_data['Goals Conceded'][i] else league_data['Result'][i] for i in range(len(league_data))]

    available_vars = ['Possession','xG','xGA','xGD','Goals','Goals Conceded','GD','GD-xGD','Shots','Shots Faced','Field Tilt','Passes in Opposition Half','Passes into Box','xT','xT Against','Shots per 1 xT','xG per 1 xT','xGA per 1 xT Against','PPDA','High Recoveries','Crosses','Corners','Fouls']
else:
    available_vars = ['Possession','Shots','Field Tilt','Passes in Opposition Half','Passes into Box','xT','Shots per 1 xT','PPDA','High Recoveries','Crosses','Corners','Fouls']

team_data[available_vars] = team_data[available_vars].astype(float)
league_data[available_vars] = league_data[available_vars].astype(float)


report_tab, data_tab, graph_tab, xg_tab = st.tabs(['Match Report', 'Data by Match - Table', 'Data by Match - Graph', 'xG & xGA by Match'])

report_tab.image(game_image)
data_tab.write(team_data)
with graph_tab:
    var = st.selectbox('Metric to Plot', available_vars)
    c = (
       alt.Chart(team_data[::-1], title=alt.Title(
       f"{team} {var}, {league}",
       subtitle=[f"Data via Opta | Created by Ben Griffis (@BeGriffis) | Data as of {update_date}","Generated on: football-match-reports.streamlit.app"]
   ))
       .mark_line(point=True)
       .encode(x=alt.X('Date', sort=None), y=var, tooltip=['Match','Date',var,'Possession']).properties(height=500)
    )
    st.altair_chart(c, use_container_width=True)

with xg_tab:
    lg_chart = alt.Chart(league_data,  title=alt.Title(
       f"{team} xG & xGA by Match, {league}",
       subtitle=[f"Data via Opta | Created by Ben Griffis (@BeGriffis) | Data as of {update_date}",f"Small grey points are all matches in the league. Large Colored points are {team}'s matches","Generated on: football-match-reports.streamlit.app"],
    )).mark_circle(size=30, color='silver').encode(
        x='xG',
        y='xGA',
        # color='Result',
        tooltip=['Team','Match','Date','xGD','Possession','Field Tilt']
    ).properties(height=500).interactive()

    domain = ['W','D','L']
    range_ = ['blue','black','darkorange']
    team_chart = alt.Chart(team_data,  title=alt.Title(
       f"{team} xG & xGA by Match, {league}",
       subtitle=[f"Data via Opta | Created by Ben Griffis (@BeGriffis) | Data as of {update_date}",f"Small grey points are all matches in the league. Large Colored points are {team}'s matches","Generated on: football-match-reports.streamlit.app"],
    )).mark_circle(size=90).encode(
        x='xG',
        y='xGA',
        color=alt.Color('Result').scale(domain=domain, range=range_),
        tooltip=['Team','Match','Date','xGD','Possession','Field Tilt']
    ).properties(height=500).interactive()

    chart = (lg_chart + team_chart)

    st.altair_chart(chart, use_container_width=True)

