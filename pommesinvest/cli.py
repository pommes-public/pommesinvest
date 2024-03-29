from pommesinvest.model.investment_model import run_investment_model, add_args


def create_default_config():
    content = """# Determine the model configuration

# 1) Set overall workflow control parameters
control_parameters:
    multi_period: True
    myopic_horizon: False
    aggregate_input: False
    interest_rate: 0.02
    countries:
    [
        "AT",
        "BE",
        "CH",
        "CZ",
        "DE",
        "DK1",
        "DK2",
        "FR",
        "NL",
        "NO1",
        "NO2",
        "NO3",
        "NO4",
        "NO5",
        "PL",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "IT",
    ]
    solver: "gurobi"
    solver_commandline_options: False
    solver_tmp_dir: "default"
    fuel_cost_pathway: "NZE"
    emissions_cost_pathway: "long-term"
    flexibility_options_scenario: "50"
    activate_emissions_budget_limit: False
    activate_emissions_pathway_limit: True
    emissions_pathway: "KNS_2035"
    use_technology_specific_wacc: True
    activate_demand_response: False
    demand_response_approach: "DLR"
    demand_response_scenario: "50"
    use_subset_of_delay_times: False
    impose_investment_maxima: True
    include_artificial_shortage_units: True
    save_production_results: True
    save_investment_results: True
    write_lp_file: False
    extract_duals: True
    extract_other_countries_production: False
    results_rounding_precision: 2
    sensitivity_parameter: "None"
    sensitivity_value: "None"

# 2) Set model optimization time and frequency
time_parameters:
    start_time: "2020-01-01 00:00:00"
    end_time: "2020-12-30 23:00:00"
    freq: "1H"

# 3) Set input and output data paths
input_output_parameters:
    path_folder_input: "./inputs/"
    path_folder_output: "./results/"

# 4) Set myopic horizon parameters (optional)
myopic_horizon_parameters:
    myopic_horizon_in_years: 4
    overlap_in_years: 0
    
# 5) Set solver command line options (optional)
solver_cmdline_options:
    lpmethod: 4
    preprocessing dual: -1
    solutiontype: 2
    threads: 12
    barrier convergetol: 1.0e-6"""
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
