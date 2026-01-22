---
language:
- en
tags:
- basketball
- nba
- sports
- tracking
- play-by-play
pretty_name: NBA 2015/2016 Season Raw Tracking Data from SportVU
source_datasets: 
- https://github.com/linouk23/NBA-Player-Movements
- https://github.com/sumitrodatta/nba-alt-awards
---
# 2015-2016 Raw Tracking Data from SportVU

The modern era of basketball is characterized by the use of data to analyze performance and make decisions both on and off the court. Using tracking data combined with traditional play-by-play can allow for in=depth analysis of games.

## Dataset Details

### Dataset Descriptions

Tracking data is the finest level of basketball data, whereas play-by-play and box score data are also used. This dataset gives raw SportVU tracking data from each game of the 2015-2016 NBA season merged with play-by-play data. 2015-16 was the last season with publically available tracking data. This data has the coordinates of all players at all moments of the game, for each game in the season. There is also more information such as descriptors for players on the team (and their unique IDs) and the teams playing (and their unique IDs). Further, descriptors of the play that occured at each event is present, and the team in possession during the event, along with more necessary features.
- **Collected By:** SportVU, Basketball Referece
- **Shared By:** Kostya Linou, Dzmitryi Linou, Martijn De Boer, Sumitro Datta

### Dataset Source

- **Repositories:**
  - https://github.com/linouk23/NBA-Player-Movements
  - https://github.com/sumitrodatta/nba-alt-awards

## Uses

This dataset has many potential uses. Primarily, visualization of plays, as illustrated in the initial repository is possible, creating a comprehensive view for analyzing actions on court. Beyond that, models could be trained to recognize certain play types or actions, which can increase efficiency of video scouting. Analysis of defensive control could be performed by examining the data spatially. Even further, a broadcast tracking model could be creater if video data could be obtained and connected to each moment of collection. This would create a model where video frames are mapped to tracked coordinates, increasing the accessibility of tracking data as only publically available video footage is necessary. 

- An example of action identification shown here: https://colab.research.google.com/drive/1x_v9c5yzUnDvSsH9d-2m3FjFXMp8A-ZF?usp=sharing

## Dataset Structure

The data is in the following dictionary format:

- 'gameid': str (ID for the game)
- 'gamedate': str (date the game occured on)
- 'event_info': 
  - 'eventid': str (ID for the event in the given game)
  - 'type': int (number corresponding to event type)
  - 'possession_team_id': float (team ID of team in possession during the event)
  - 'desc_home': str (description of the event for the home team)
  - 'desc_away': str (description of the event for the away team)
- 'primary_info':
  - 'team': str (home or visitor)
  - 'player_id': float (ID of primary player involved in event)
  - 'team_id': float (ID of team for primary player)
- 'secondary_info': same format as primary info, but for a secondary player involved
- 'visitor':
  - 'name': str (team name)
  - 'teamid': int (team ID)
  - 'abbreviation': str (abbreviation of team name)
  - 'players': list of the dictionaries in the form of the following
    - 'lastname': str (player last name)
    - 'firstname': str (player first name)
    - 'playerid': str (player ID)
    - 'number': int (player jersey number)
    - 'position': str (player in-game position)
- 'home': same format as visitor
- 'moments': list of dictionaries in the form of the following
  - 'quarter': int (quarter of moment)
  - 'game_clock': float (game clock (seconds, descending starting from 720))
  - 'shot_clock': float (shot clock (seconds, descending starting from 24))
  - 'ball_coordinates':
    - 'x': float (x coordinate of ball)
    - 'y': float (y coordinate of ball)
    - 'z': float (z coordinate of ball)
  - 'player_coordinates': list of the dictionaries in the form of the following, 
    - 'teamid': int (team ID of player)
    - 'playerid': int (player ID for player)
    - 'x': float (x coordinate of player)
    - 'y': float (y coordinate of player)
    - 'z': float (z coordinate of player)

## Requirements

To load the data, you must run

`import py7zr`

## Configurations

The data here has multiple configurations corresponding to different size subsamples of the data. This is intended for quicker loading and increased manageability. The configurations are as follows:
- 'tiny': a subsample of 5 games
- 'small': a subsample of 25 games
- 'medium': a subsample of 100 games
- 'large': all games (600+) with tracking data from 2015-16 NBA season


## Dataset Creation

### Curation Rationale

The reason for uploading this data to huggingface, is that in its current .7z form, the data is less accessible, and requires unzipping many files and then combining to access. Also, more sources for easily accessible tracking data, even if also available elsewhere, increase the chances of long-term preservation and accessibility for future NBA fans.

On top of that, tracking data combined with play-by-play data is ideal format of sports data, as there is little confusion and allows for better labeling of events.

### Source Data

From creator StatsPerform, "the SportVU camera system is installed in basketball arenas to track the real-time positions of players and the ball at 25 times per second." These methods were used to capture the data in this dataset.

## Bias, Risks, and Limitations

Technical limitations include the following: 

Some events or moments included within events have no corresponding coordinates, which can cause trouble with continuity, however this is not a major problem as this only occurs on a very small number of events and the occurances can be handled on a case-by-case basis or ignored.

The coordinates for each event often start before the labeled event and/or end after the event ends. This can also cause bleeding of data over to the next event, so care must be taken to acknowledge this when working with the data.

Since this data is not up-to-date, and the tracking data for the last eight seasons is private and unreleased, the continued spread of this specific data may not be representative of the current state of NBA tracking data (provided by different companies). Thus, users that learn how to manipulate it may or may not be adequately prepared for work in basketball organizations. 

Further, analyses performed on the dataset may not be reflective of the current state of professional basketball. This is because the game is constantly changing and evolving. However, since this was the last iteration of publicly available tracking data, I believe increasing its availability is important.

## Dataset Card Author

Donald Cayton; dcayton9@gmail.com