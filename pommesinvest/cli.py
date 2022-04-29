from pommesinvest.model.investment_model import run_investment_model, add_args


def create_default_config():
    content = """# Determine the model configuration

# 1) Set overall workflow control parameters
control_parameters:
    rolling_horizon: False
    aggregate_input: False
    interest_rate: 0.02
    solver: "gurobi"
    fuel_cost_pathway: "NZE"
    emissions_cost_pathway: "long-term"
    investment_cost_pathway: "middle"
    activate_emissions_limit: False
    emissions_pathway: "100_percent_linear"
    activate_demand_response: False
    demand_response_approach: "DLR"
    demand_response_scenario: "50"
    impose_investment_maxima: True
    save_production_results: True
    save_investment_results: True
    write_lp_file: False

# 2) Set model optimization time and frequency
time_parameters:
    start_time: "2017-01-01 00:00:00"
    end_time: "2017-01-02 23:00:00"
    freq: "4H"

# 3) Set input and output data paths
input_output_parameters:
    path_folder_input: "./inputs/"
    path_folder_output: "./results/"

# 4) Set myopic horizon parameters (optional)
myopic_horizon_parameters:
    myopic_horizon_in_years: 4
    overlap_in_years: 0"""
    with open("./config.yml", "w") as opf:
        opf.write(content)


def run_pommes_invest():
    args = add_args()
    if args.init:
        create_default_config()
        return
    run_investment_model(args.file)


if __name__ == "__main__":
    run_pommes_invest()
