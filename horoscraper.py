import re
import urllib2
import time
import json
from datetime import datetime

from dateutil import parser
from dateutil import rrule
import pandas as pd


signs = ["general", "aries", "taurus", "gemini", "cancer", "leo", "virgo",
         "libra", "scorpio", "saggitarius", "capricorn", "aquarius", "picies"]
horoscope_url = "http://www.tarot.com/daily-horoscope/general/{date}"


def extract_horoscopes_from_page(page):
    # Extract the horoscopes from the page. Tarot.com sometimes formats the
    # page differently, so we need to try the extraction in two ways.
    horoscope_json_fmt1 = re.search("window\.horoscopes = (\[{.+?}\]);", page)
    horoscope_json_fmt2 = re.findall("\"\d{1,}\":({.+?})", page)
    if horoscope_json_fmt1:
        horoscope_df = pd.DataFrame(json.loads(horoscope_json_fmt1.group(1)))
    elif horoscope_json_fmt2:
        horoscope_df = pd.DataFrame((json.loads(j) for j in horoscope_json_fmt2))
    else:
        print("Cannot find horoscopes on page.")
        return pd.DataFrame()

    # Map tarot.com astrological sign id numbers to each sign name
    for sign_id, sign_name in enumerate(signs):
        horoscope_df.sign[ horoscope_df.sign == str(sign_id) ] = sign_name

    # Parse dates/numbers & get rid of unneeded columns
    horoscope_df.date = horoscope_df.date.apply(parser.parse)
    horoscope_df.rating = horoscope_df.rating.apply(int)
    horoscope_df.drop("image", axis=1, inplace=True)
    horoscope_df.drop("slant", axis=1, inplace=True)
    return horoscope_df


def fetch_horoscopes(start_date, end_date, data_filename):
    # Write the column names of our data table to a csv first
    data_col_names = ["date", "full_text", "keywords", "stars", "sign",
                      "sms_summary", "subject_line"]
    pd.DataFrame(columns=data_col_names).to_csv(data_filename, index=False)

    # Fetch all horoscopes for each day and append them to the csv
    for day in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        date_str = day.strftime("%Y-%m-%d")
        print("Fetching horoscopes for %s" % date_str)
        page = urllib2.urlopen(horoscope_url.format(date=date_str)).read()

        horoscopes = extract_horoscopes_from_page(page)
        horoscopes.to_csv(data_filename, mode="a", header=False, index=False)

        # Politely wait a bit after each request
        time.sleep(1)
    return


def main():
    # Fetch all daily horoscopes from tarot.com from Jan 1, 2008 to today.
    # Save the horoscopes to a csv so that we can play around with them in
    # pandas/scikit-learn.
    fetch_horoscopes(start_date=datetime(2008, 1, 1),
                          end_date=datetime.now(),
                          data_filename="data.csv")
    return


if __name__ == "__main__":
    main()
