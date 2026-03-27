import pandas as pd


def main():


    pv_df = pd.read_csv("utrecht/pv_data/pv_hourly_2014_2017.csv")
    weather_df = pd.read_csv("utrecht/knmi_data/knmi_hourly_2014_2017.csv")


    pv_df = pv_df[["pv_total_kWh"]]


    merged_pv_data = weather_df.merge(
        pv_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    merged_pv_data.to_csv("utrecht/processed_data/utrecht_pv_data.csv",index=False)



if __name__ == "__main__":
    main()