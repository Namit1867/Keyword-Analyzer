def keywordPlanner(condition):
    keyword_planner = """
Instruction:
Hey there! As a digital marketer, you have a task at hand to optimize your Google Ads campaign using data from Google Keyword Planner. You'll be working with a large Excel sheet DataFrame that contains keywords, their relevant ad groups, and several metrics such as Low and High Bids on Keywords, Average Searches, Competition Level, and Competition Index. Your main objectives are twofold:

1. Identify the Best-Performing Keywords: Analyze the data to identify the keywords that show the most promise in terms of relevance with given ad groups & performance. Focus on metrics like Average Searches, Low and High Bids on Keywords, and Competition Level. These factors will help you determine the keywords that are both relevant to your campaign and likely to attract a substantial audience.

2. Allocate a Daily Budget: Based on the selected keywords, you need to allocate a total daily budget for the entire campaign. The budget should be reasonable and in line with your campaign objectives. Consider the competitiveness of the keywords, the potential reach, and the specific goals of your campaign.

Note -
1. Make sure selected keywords in relevance with the given ad groups. If you do not find any keyword in relevance of the ad groups just return the message "There are no keywords in relevance to the given ad groups".
    """

    switcher = {
        1: keyword_planner,
        2: "This is response for condition 2",
        3: "This is response for condition 3"
    }
    return switcher.get(condition, "Invalid condition")
