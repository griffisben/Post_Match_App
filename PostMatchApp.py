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
team_data = team_data[team_data.Team==team].reset_index(drop=True)
team_data['Shots per 1.0 xT'] = team_data['Shots per 1.0 xT'].astype(float)
team_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)

if league in ['Saudi Pro League','Eredivisie']:
    team_data['xG per 1 xT'] = team_data['xG']/team_data['xT']
    team_data['xGA per 1 xT Against'] = team_data['xGA']/team_data['xT Against']
    available_vars = ['Possession','xG','xGA','xGD','Goals','Goals Conceded','GD','GD-xGD','Shots','Shots Faced','Field Tilt','Passes in Opposition Half','Passes into Box','xT','xT Against','Shots per 1 xT','xG per 1 xT','xGA per 1 xT Against','PPDA','High Recoveries','Crosses','Corners','Fouls']
else:
    available_vars = ['Possession','Shots','Field Tilt','Passes in Opposition Half','Passes into Box','xT','Shots per 1 xT','PPDA','High Recoveries','Crosses','Corners','Fouls']

team_data[available_vars] = team_data[available_vars].astype(float)


report_tab, data_tab, graph_tab = st.tabs(['Match Report', 'Data by Match - Table', 'Data by Match - Graph'])

report_tab.image(game_image)
data_tab.write(team_data)
with graph_tab:
    var = st.selectbox('Metric to Plot', available_vars)
    c = (
       alt.Chart(team_data[::-1], title=alt.Title(
       f"{team} {var}, {league}",
       subtitle=[f"Data via Opta | Created by Ben Griffis (@BeGriffis) | Data as of {update_date}","Generated on: football-match-reports.streamlit.app"]
   ))
       .mark_line()
       .encode(x=alt.X('Date', sort=None), y=var, tooltip=['Match','Date',var,'Possession'])
    )
    st.altair_chart(c, use_container_width=True)
