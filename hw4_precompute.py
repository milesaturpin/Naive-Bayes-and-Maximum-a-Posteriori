from __future__ import with_statement
import csv
import time
from collections import defaultdict

#load games and extract teams
teams = set()
games = []
with open("2017 NCAAM Game Results Data.csv","r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        teams.add(row['Team'])
        teams.add(row['Opponent'])
        games.append(row)


statvars = ["Score"]

#process types into native types (ints and dates)
typeConverters = dict([(v,int) for v in ["Team Score","Opponent Score","Team Margin"]])
typeConverters["Date"] = lambda(x):time.strptime(x,"%m/%d/%Y")
typeConverters["Team Differential"] = lambda(x):float(x) if len(x.strip())>0 else None
typeConverters["Opponent Differential"] = lambda(x):float(x) if len(x.strip())>0 else None
for game in games:
    for (k,v) in typeConverters.iteritems():
        game[k] = v(game[k])

#remove duplicates with where team vs opponent is flipped
uniquegames = []
gamesbydate = defaultdict(list)
uniquegamesbydate = dict()
for g in games:
    gamesbydate[g["Date"]].append(g)
for date,dgames in sorted(gamesbydate.items()):
    unique = []
    for g in dgames:
        if all(g["Team"]!=h["Opponent"] for h in unique):
            unique.append(g)
    uniquegames += unique
    uniquegamesbydate[date] = unique
print len(games),"records,",len(uniquegames),"unique"

gamesbyteam = defaultdict(list)
for game in uniquegames:
    gamesbyteam[game['Team']].append(game)
    gamesbyteam[game['Opponent']].append(game)

def inGame(team,game):
    """True if the team was in the given game"""
    return team==game["Team"] or team==game["Opponent"]

def atHome(team,game):
    """True if the team was at home in the given game"""
    return (team==game["Team"] and 'Home'==game["Team Location"]) or \
            (team==game["Opponent"] and 'Away'==game["Opponent Location"])

def atOpponent(team,game):
    """True if the team was at opponent's home the given game"""
    return (team==game["Team"] and 'Away'==game["Team Location"]) or \
            (team==game["Opponent"] and 'Home'==game["Opponent Location"])

def wonGame(team,game):
    if game["Team Score"] > game["Opponent Score"]:
        return team==game["Team"]
    else:
        return team==game["Opponent"]

def gamesBefore(date,team,games):
    """Returns the set of games where the team participated before the given date"""
    res = []
    for g in gamesbyteam[team]:
        if g["Date"]<date:
            res.append(g)
    return res

def averageGained(games,team,item):
    """Returns the averagetotals of the given item over the games"""
    ingames = [g for g in games if inGame(team,g)]
    if len(ingames)==0: return 0
    sumvalues = 0
    for g in ingames:
        if team==g["Team"]:
            sumvalues += g["Team "+item]
        else:
            sumvalues += g["Opponent "+item]
    return float(sumvalues) / len(ingames)

def averageAllowed(games,team,item):
    """Returns the average totals of the given team's opponent for the given item"""
    ingames = [g for g in games if inGame(team,g)]
    if len(ingames)==0: return 0
    sumvalues = 0
    for g in ingames:
        if team==g["Team"]:
            sumvalues += g["Opponent "+item]
        else:
            sumvalues += g["Team "+item]
    return float(sumvalues) / len(ingames)

def record(team,games):
    """Returns a pair (wins,losses) for the given team in the given set of games"""
    ingames = [g for g in games if inGame(team,g)]
    wins = len([g for g in ingames if wonGame(team,g)])
    return (wins,len(ingames)-wins)

def make_features(game,games):
    team = game["Team"]
    opponent = game["Opponent"]
    #extract history of team and opponent
    teamhistory = gamesBefore(game["Date"],team,games)
    opphistory = gamesBefore(game["Date"],opponent,games)
    
    #compute the win/loss record
    teamrecord = record(team,teamhistory)
    opprecord = record(opponent,opphistory)
    
    #compute average statistics for team and opponent
    teamavgstats = {}
    for s in statvars:
        teamavgstats[s] = averageGained(teamhistory,team,s)
        teamavgstats[s+"Allowed"] = averageAllowed(teamhistory,team,s)
    oppavgstats = {}
    for s in statvars:
        oppavgstats[s] = averageGained(opphistory,opponent,s)
        oppavgstats[s+"Allowed"] = averageAllowed(opphistory,opponent,s)
        
    #start building feature dictionary
    features = dict()
    features["Date"] = time.strftime("%m/%d/%Y",game["Date"])
    for k in ["Team","Opponent","Team Differential","Opponent Differential"]:
        features[k] = game[k]
    features["at_home"] = 1 if atHome(team,game) else 0
    features["at_opp"] = 1 if atOpponent(team,game) else 0
    features["team_wins"]=teamrecord[0]
    features["team_losses"]=teamrecord[1]
    for (k,v) in teamavgstats.iteritems():
        features["team_avg_"+k] = v
    features["opp_wins"]=opprecord[0]
    features["opp_losses"]=opprecord[1]
    for (k,v) in oppavgstats.iteritems():
        features["opp_avg_"+k] = v
    features["team_won"] = 1 if wonGame(team,game) else 0
    return features

def make_all_features(games):
    """Outputs a feature dictionary [feature_dict(g)] for the given games"""
    return [make_features(g,games) for g in games]

#build the feature list
features = make_all_features(uniquegames)

outfn = "2017 NCAAM Game Results Features.csv"
print "Saving to",outfn
with open(outfn,"w") as csvfile:
    writer = csv.DictWriter(csvfile,sorted(features[0].keys()))
    writer.writeheader()
    for f in features:
        writer.writerow(f)

