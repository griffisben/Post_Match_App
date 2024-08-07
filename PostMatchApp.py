import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

def complementaryColor(my_hex):
    """
    https://stackoverflow.com/questions/38478409/finding-out-complementary-opposite-color-of-a-given-color
    """
    if my_hex[0] == '#':
        my_hex = my_hex[1:]
    rgb = (my_hex[0:2], my_hex[2:4], my_hex[4:6])
    comp = ['%02X' % (255 - int(a, 16)) for a in rgb]
    return ''.join(comp)

lg_lookup = pd.read_csv("https://raw.githubusercontent.com/griffisben/Post_Match_App/main/PostMatchLeagues.csv")
league_list = lg_lookup.League.tolist()

with st.sidebar:
    league = st.selectbox('What League Do You Want Reports For?', league_list, index=league_list.index('Ekstraklasa'))
    update_date = lg_lookup[lg_lookup.League==league].Update.values[0]
    
st.title(f"{league} Post-Match Reports")
st.subheader(f"Last Updated: {update_date}\n")
st.subheader('All data via Opta')

df = pd.read_csv(f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/League_Files/{league.replace(' ','%20')}%20Full%20Match%20List.csv")
df['Match_Name'] = df['Match'] + ' ' + df['Date']

with st.sidebar:
    team_list = sorted(list(set(df.Home.unique().tolist() + df.Away.unique().tolist())))
    team = st.selectbox('What team do you want reports & data for?', team_list)

    specific = st.selectbox('Specific Match or Most Recent Matches?', ('Specific Match','Recent Matches'))
    if specific == 'Specific Match':
        match_list = df[(df.Home == team) | (df.Away == team)].copy()
        match_choice = st.selectbox('Match', match_list.Match_Name.tolist())
        render_matches = [match_choice]
    if specific == 'Recent Matches':
        match_list = df[(df.Home == team) | (df.Away == team)].copy()
        num_matches = st.slider('Number of Recent Matches', min_value=1, max_value=5, value=3)
        render_matches = match_list.head(num_matches).Match_Name.tolist()

    focal_color = st.color_picker("Pick a color to highlight the team on League Ranking tab", "#4c94f6")

#########################
def ben_theme():
    return {
        'config': {
            'background': '#fbf9f4',
            # 'text': '#4a2e19',
            # 'mark': {
            #     'color': focal_color,
            # },
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

report_tab, data_tab, graph_tab, rank_tab = st.tabs(['Match Report', 'Data by Match - Table', 'Data by Match - Graph', 'League Rankings'])

if league not in ['Ekstraklasa 23-24']:
    for i in range(len(render_matches)):
        match_string = render_matches[i].replace(' ','%20')
        url = f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/Image_Files/{league.replace(' ','%20')}/{match_string}.png"
        response = requests.get(url)
        game_image = Image.open(io.BytesIO(response.content))
        report_tab.image(game_image)
else:
    with report_tab:
        st.write("Sorry, I don't have post-match reports loaded for this league!")

team_data = pd.read_csv(f"https://raw.githubusercontent.com/griffisben/Post_Match_App/main/Stat_Files/{league.replace(' ','%20')}.csv")
team_data['Field Tilt - Possession'] = team_data['Field Tilt'] - team_data['Possession']
team_data['xT Difference'] = team_data['xT'] - team_data['xT Against']

league_data = team_data.copy().reset_index(drop=True)
team_data = team_data[team_data.Team==team].reset_index(drop=True)
if league in ['Ekstraklasa 23-24']:
    team_data['Shots Faced per 1.0 xT Against'] = team_data['Shots Faced']/team_data['xT Against']
    league_data['Shots Faced per 1.0 xT Against'] = league_data['Shots Faced']/league_data['xT Against']

team_data['Shots per 1.0 xT'] = team_data['Shots per 1.0 xT'].astype(float)
team_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)
team_data['Shots Faced per 1.0 xT Against'] = team_data['Shots Faced per 1.0 xT Against'].astype(float)
team_data.rename(columns={'Shots Faced per 1.0 xT Against':'Shots Faced per 1 xT Against'},inplace=True)

league_data['Shots per 1.0 xT'] = league_data['Shots per 1.0 xT'].astype(float)
league_data.rename(columns={'Shots per 1.0 xT':'Shots per 1 xT'},inplace=True)
league_data['Shots Faced per 1.0 xT Against'] = league_data['Shots Faced per 1.0 xT Against'].astype(float)
league_data.rename(columns={'Shots Faced per 1.0 xT Against':'Shots Faced per 1 xT Against'},inplace=True)


team_data['xG per 1 xT'] = team_data['xG']/team_data['xT']
league_data['xG per 1 xT'] = league_data['xG']/league_data['xT']

team_data['xGA per 1 xT Against'] = team_data['xGA']/team_data['xT Against']
league_data['xGA per 1 xT Against'] = league_data['xGA']/team_data['xT Against']

available_vars = ['Possession',
                  # 'xG','xGA','xGD',
                  'Goals','Goals Conceded',
                  # 'GD','GD-xGD',
                  'Shots','Shots Faced','Field Tilt','Field Tilt - Possession','Avg Pass Height','Passes in Opposition Half','Passes into Box','xT','xT Against','xT Difference','Shots per 1 xT','Shots Faced per 1 xT Against',
                  # 'xG per 1 xT','xGA per 1 xT Against',
                  'PPDA','High Recoveries','High Recoveries Against','Crosses','Corners','Fouls',
                 'Throw-Ins into the Box','On-Ball Pressure','On-Ball Pressure Share','Off-Ball Pressure','Off-Ball Pressure Share','Game Control','Game Control Share',
                 ]

team_data[available_vars] = team_data[available_vars].astype(float)
league_data[available_vars] = league_data[available_vars].astype(float)

league_data_base = league_data.copy()

data_tab.write(team_data)

with graph_tab:
    plot_type = st.radio("Line or Bar plot?", ['📈 Line', '📊 Bar'])
    var = st.selectbox('Metric to Plot', available_vars)

    if plot_type == '📈 Line':
        lg_avg_var = league_data[var].mean()
        team_avg_var = team_data[var].mean()
        
        c = (alt.Chart(
                team_data[::-1],
                title={
                    "text": [f"{team} {var}, {league}"],
                    "subtitle": [f"Data via Opta as of {update_date} | Created: Ben Griffis (@BeGriffis) via football-match-reports.streamlit.app"]
                }
            )
            .mark_line(point=True, color='#4c94f6')
            .encode(
                x=alt.X('Date', sort=None),
                y=alt.Y(var, scale=alt.Scale(zero=False)),
                tooltip=['Match', 'Date', var, 'Possession','Field Tilt']
            )
        )
    
        lg_avg_line = alt.Chart(pd.DataFrame({'y': [lg_avg_var]})).mark_rule(color='#ee5454').encode(y='y')
        
        lg_avg_label = lg_avg_line.mark_text(
            x="width",
            dx=-2,
            align="right",
            baseline="bottom",
            text="League Avg",
            color='#ee5454'
        )
    
        team_avg_line = alt.Chart(pd.DataFrame({'y': [team_avg_var]})).mark_rule(color='#f6ba00').encode(y='y')
        
        team_avg_label = team_avg_line.mark_text(
            x="width",
            dx=-2,
            align="right",
            baseline="bottom",
            text="Team Avg",
            color='#f6ba00'
        )
    
    
        chart = (c + lg_avg_line + lg_avg_label + team_avg_line + team_avg_label)
        st.altair_chart(chart, use_container_width=True)

    if plot_type == '📊 Bar':
        lg_avg_var = league_data[var].mean()
        team_avg_var = team_data[var].mean()

        c = (alt.Chart(
                team_data[::-1],
                title={
                    "text": [f"{team} {var}, {league}"],
                    "subtitle": [f"Data via Opta as of {update_date} | Created: Ben Griffis (@BeGriffis) via football-match-reports.streamlit.app"]
                }
            )
            .mark_bar()
            .encode(
                x=alt.X('Date', sort=None),
                y=alt.Y(var, scale=alt.Scale(zero=False)), 
                color=alt.condition(alt.datum[var] >= 0, alt.value('#4c94f6'), alt.value('#4a2e19')),
                tooltip=['Match', 'Date', var, 'Possession','Field Tilt']
            )
        )

        if var != 'xT Difference':
            lg_avg_line = alt.Chart(pd.DataFrame({'y': [lg_avg_var]})).mark_rule(color='#ee5454').encode(y='y')
            
            lg_avg_label = lg_avg_line.mark_text(
                x="width",
                dx=-2,
                align="right",
                baseline="bottom",
                text="League Avg",
                color='#ee5454'
            )
        if var == 'xT Difference':
            lg_avg_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='k').encode(y='y')
    
        team_avg_line = alt.Chart(pd.DataFrame({'y': [team_avg_var]})).mark_rule(color='#f6ba00').encode(y='y')
        
        team_avg_label = team_avg_line.mark_text(
            x="width",
            dx=-2,
            align="right",
            baseline="bottom",
            text="Team Avg",
            color='#f6ba00'
        )
    

        if var != 'xT Difference':
            chart = (c + lg_avg_line + lg_avg_label + team_avg_line + team_avg_label)
        if var == 'xT Difference':
            chart = (c + lg_avg_line + team_avg_line + team_avg_label)
        st.altair_chart(chart, use_container_width=True)


with rank_tab:
    ranking_base_df = league_data_base.copy()
    rank_method = st.selectbox('Ranking Method', ['Average','Total','Median'])
    rank_var = st.selectbox('Metric to Rank', available_vars)

    if rank_method == 'Median':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].median().reset_index()
    if rank_method == 'Total':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].sum().reset_index()
    if rank_method == 'Average':
        rank_df = ranking_base_df.groupby(['Team'])[available_vars].mean().reset_index()

    if rank_var in ['xGA','Goals Conceded','Shots Faced','xT Against','xGA per 1 xT Against','PPDA','Fouls','High Recoveries Against', 'Shots Faced per 1 xT Against']:
        sort_method = True
    else:
        sort_method = False

    indexdf_short = rank_df.sort_values(by=[rank_var],ascending=sort_method)[['Team',rank_var]].reset_index(drop=True)[::-1]
    
    sns.set(rc={'axes.facecolor':'#fbf9f4', 'figure.facecolor':'#fbf9f4',
           'ytick.labelcolor':'#4A2E19', 'xtick.labelcolor':'#4A2E19'})

    fig = plt.figure(figsize=(7,8), dpi=200)
    ax = plt.subplot()
    
    ncols = len(indexdf_short.columns.tolist())+1
    nrows = indexdf_short.shape[0]

    ax.set_xlim(0, ncols + .5)
    ax.set_ylim(0, nrows + 1.5)
    
    positions = [0.05, 2.0]
    columns = indexdf_short.columns.tolist()
    
    # Add table's main text
    for i in range(nrows):
        for j, column in enumerate(columns):
            if column == 'Team':
                if nrows-i < 10:
                    text_label = f'{nrows-i}     {indexdf_short[column].iloc[i]}'
                else:
                    text_label = f'{nrows-i}   {indexdf_short[column].iloc[i]}'
            else:
                text_label = f'{round(indexdf_short[column].iloc[i],2)}'
            if indexdf_short['Team'].iloc[i] == team:
                t_color = focal_color
                weight = 'bold'
            else:
                t_color = '#4A2E19'
                weight = 'regular'
            ax.annotate(
                xy=(positions[j], i + .5),
                text = text_label,
                ha='left',
                va='center', color=t_color,
                weight=weight
            )
            
    # Add column names
    column_names = columns
    for index, cs in enumerate(column_names):
            ax.annotate(
                xy=(positions[index], nrows + .25),
                text=column_names[index],
                ha='left',
                va='bottom',
                weight='bold', color='#4A2E19'
            )

    # Add dividing lines
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
    for x in range(1, nrows):
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')
    
    ax.set_axis_off()
    
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # -- Take data coordinates and transform them to normalized figure coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
    # -- Add nation axes
    ax_point_1 = DC_to_NFC([2.25, 0.25])
    ax_point_2 = DC_to_NFC([2.75, 0.75])
    ax_width = abs(ax_point_1[0] - ax_point_2[0])
    ax_height = abs(ax_point_1[1] - ax_point_2[1])

    fig.text(
        x=0.14, y=.91,
        s=f"{rank_var} {rank_method} Rankings",
        ha='left',
        va='bottom',
        weight='bold',
        size=13, color='#4A2E19'
    )
    fig.text(
        x=0.14, y=.9,
        s=f"Data via Opta as of {update_date}  \nCreated: Ben Griffis (@BeGriffis) via football-match-reports.streamlit.app",
        ha='left',
        va='top',
        weight='regular',
        size=11, color='#4A2E19'
    )

    fig
