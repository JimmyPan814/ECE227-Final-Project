"""Underlying simulator for the YYG/C19Pro SEIR model.

Learn more at: https://github.com/youyanggu/yyg-seir-simulator. Developed by Youyang Gu.
"""

import datetime

import numpy as np

from fixed_params import *


def get_daily_imports(region_model, i):
    """Returns the number of new daily imported cases based on day index i (out of N days).

    - beginning_days_flat is how many days at the beginning we maintain a constant import.
    - end_days_offset is the number of days from the end of the projections
        before we get 0 new imports.
    - The number of daily imports is initially region_model.daily_imports, and
        decreases linearly until day N-end_days_offset.
    """

    N = region_model.N
    assert i < N, 'day index must be less than total days'

    if hasattr(region_model, 'beginning_days_flat'):
        beginning_days_flat = region_model.beginning_days_flat
    else:
        beginning_days_flat = 10
    assert beginning_days_flat >= 0

    if hasattr(region_model, 'end_days_offset'):
        end_days_offset = region_model.end_days_offset
    else:
        end_days_offset = int(N - min(N, DAYS_WITH_IMPORTS))
    assert beginning_days_flat + end_days_offset <= N
    n_ = N - beginning_days_flat - end_days_offset + 1

    daily_imports = region_model.daily_imports * \
        (1 - min(1, max(0, (i-beginning_days_flat+1)) / n_))

    if region_model.country_str not in ['China', 'South Korea', 'Australia'] and not \
            hasattr(region_model, 'end_days_offset'):
        # we want to maintain ~10 min daily imports a day
        daily_imports = max(daily_imports, min(10, 0.1 * region_model.daily_imports))

    return daily_imports 


def run(region_model):
    """Given a RegionModel object, runs the SEIR simulation."""
    dates = np.array([region_model.first_date + datetime.timedelta(days=i) \
        for i in range(region_model.N)])
    infections = np.array([0.] * region_model.N)
    hospitalizations = np.zeros(region_model.N) * np.nan
    deaths = np.array([0.] * region_model.N)
    reported_deaths = np.array([0.] * region_model.N)
    mortaility_rates = np.array([region_model.MORTALITY_RATE] * region_model.N) 

    assert infections.dtype == hospitalizations.dtype == \
        deaths.dtype == reported_deaths.dtype == mortaility_rates.dtype == np.float64

    """
    We compute a normalized version of the infections and deaths probability distribution.
    We invert the infections and deaths norm to simplify the convolutions we will take later.
        Aka the beginning of the array is the farther days out in the convolution.
    """
    deaths_norm = DEATHS_DAYS_ARR[::-1] / DEATHS_DAYS_ARR.sum()
    infections_norm = INFECTIOUS_DAYS_ARR[::-1] / INFECTIOUS_DAYS_ARR.sum() 

    ## Simulate effect of quarantine 
    # print("infections_norm before", infections_norm)
    if hasattr(region_model, 'quarantine_fraction'):
        # reduce infections in the latter end of the infectious period, based on reduction_idx
        infections_norm[:region_model.reduction_idx] = \
            infections_norm[:region_model.reduction_idx] * (1 - region_model.quarantine_fraction)
        infections_norm[region_model.reduction_idx] = \
            (infections_norm[region_model.reduction_idx] * 0.5) + \
            (infections_norm[region_model.reduction_idx] * 0.5 * \
                (1 - region_model.quarantine_fraction))
    # print("deaths_norm", deaths_norm)
    # print("infections_norm after", infections_norm)
    # exit()
    # the greater the immunity mult, the greater the effect of immunity
    assert 0 <= region_model.immunity_mult <= 2, region_model.immunity_mult

    ########################################
    # Compute infections
    ########################################
    effective_r_arr = []
    r_immunity_perc_arr = []
    total_immune_proportion_arr = []
    perc_population_infected_thus_far_arr = []
    vaccinated_proportion_arr = [] 
    # print("R_0_ARR", region_model.R_0_ARR)
    # exit()

    # print("INCUBATION_DAYS", INCUBATION_DAYS)
    # exit()

    # simulate the effect of vaccination 
    vaccinations = region_model.vaccination_coverage_arr
    vaccine_efficacy = region_model.vaccine_efficacy

    for i in range(region_model.N):
        if i < INCUBATION_DAYS+len(infections_norm):
            # initialize infections
            infections[i] = region_model.daily_imports
            effective_r_arr.append(region_model.R_0_ARR[i])
            r_immunity_perc_arr.append(0)
            total_immune_proportion_arr.append(0)
            perc_population_infected_thus_far_arr.append(0)
            vaccinated_proportion_arr.append(0)
            continue

        # assume 50% of population lose immunity after 6 months
        infected_thus_far = infections[:max(0, i-180)].sum() * 0.5 + infections[max(0, i-180):i-1].sum()
        perc_population_infected_thus_far = \
            min(1., infected_thus_far / region_model.population)
        assert 0 <= perc_population_infected_thus_far <= 1, perc_population_infected_thus_far 

        if region_model.skip_vaccinations:  
            vaccinated_proportion = 0.0
        else: 
            ## 1st attempt to add the effect of vaccinations
            # assume 50% of population lose immunity after 6 months? 
            # vaccinated_thus_far = vaccinations[:i].sum()
            # vaccinated_thus_far = vaccinations[:max(0, i-180)].sum() * 0.1 + vaccinations[max(0, i-180):i-1].sum()  
            vaccinated_thus_far = region_model.total_vaccination_arr[i]

            vaccinated_proportion = vaccinated_thus_far * vaccine_efficacy / region_model.population 
             
            ## 1st attempt end 

            ## 2nd attempt to add the effect of vaccinations (official method)
            # natural_infection_rate = 0.35 # US: 30~40%  
            # vaccination_rate_total_population = 0.55 # US: ~55% 
            # natural_infection_rate = infections[:i].sum() / region_model.population 
            # vaccination_rate_total_population = vaccinations[:i].sum() / region_model.population 
            # vaccination_rate_among_prev_infected = 0.45 # US: ~45% 
            # avg_vaccine_efficacy = 0.85 
            # avg_natural_immunity_efficacy = 0.85
            # avg_vaccine_and_natural_immunity_efficacy = 0.95 

            # vaccination_rate_among_no_prev_infection = \
            #     compute_vaccination_rate_among_no_prev_infection(natural_infection_rate, vaccination_rate_total_population, vaccination_rate_among_prev_infected)
            # vaccinated_only_perc = (1 - natural_infection_rate) * vaccination_rate_among_no_prev_infection
            # past_infection_only_perc = natural_infection_rate * (1 - vaccination_rate_total_population) 
            # both_vaccinated_and_past_infection_perc = natural_infection_rate * vaccination_rate_total_population

            # immunity_via_vaccination_only = vaccinated_only_perc * avg_vaccine_efficacy 
            # immunity_via_natural_infection_only = past_infection_only_perc * avg_natural_immunity_efficacy 
            # immunity_via_vaccination_and_natural_infection = both_vaccinated_and_past_infection_perc * avg_vaccine_and_natural_immunity_efficacy 

            # total_immune_proportion = immunity_via_vaccination_only + \
            #                           immunity_via_natural_infection_only + \
            #                           immunity_via_vaccination_and_natural_infection 
            # total_immune_proportion = min (1., total_immune_proportion)
            ## 2nd attempt end   
        total_immune_proportion = min(1., perc_population_infected_thus_far + vaccinated_proportion) 
        r_immunity_perc = (1. - total_immune_proportion)**region_model.immunity_mult
        effective_r = region_model.R_0_ARR[i] * r_immunity_perc
        # we apply a convolution on the infections norm array
        s = (infections[i-INCUBATION_DAYS-len(infections_norm)+1:i-INCUBATION_DAYS+1] * \
            infections_norm).sum() * effective_r
        infections[i] = s + get_daily_imports(region_model, i)

        # save intermediate computations 
        effective_r_arr.append(effective_r) 
        r_immunity_perc_arr.append(r_immunity_perc)
        total_immune_proportion_arr.append(total_immune_proportion)
        perc_population_infected_thus_far_arr.append(perc_population_infected_thus_far)
        vaccinated_proportion_arr.append(vaccinated_proportion)

    region_model.perc_population_infected_final = perc_population_infected_thus_far
    region_model.effective_r_arr = effective_r_arr 
    region_model.r_immunity_perc_arr = r_immunity_perc_arr 
    region_model.total_immune_proportion_arr = total_immune_proportion_arr
    region_model.perc_population_infected_thus_far_arr = perc_population_infected_thus_far_arr
    region_model.vaccinated_proportion_arr = vaccinated_proportion_arr 

    assert len(region_model.R_0_ARR) == len(effective_r_arr) == region_model.N
    assert len(region_model.effective_r_arr) == len(effective_r_arr) == region_model.N
    assert len(region_model.r_immunity_perc_arr) == len(effective_r_arr) == region_model.N
    assert len(region_model.total_immune_proportion_arr) == len(effective_r_arr) == region_model.N
    assert len(region_model.perc_population_infected_thus_far_arr) == len(effective_r_arr) == region_model.N
    assert len(region_model.vaccinated_proportion_arr) == len(effective_r_arr) == region_model.N
    
    # print(len(region_model.effective_r_arr))
    # print(len(region_model.r_immunity_perc_arr))
    # print(len(region_model.total_immune_proportion_arr))
    # print(len(region_model.perc_population_infected_thus_far_arr))
    # print(len(region_model.vaccinated_proportion_arr))
    # exit()

    ########################################
    # Compute hospitalizations
    ########################################
    if region_model.compute_hospitalizations:
        """
        Simple estimation of hospitalizations by taking the sum of a
            window of n days of new infections * hospitalization rate
        Note: this represents hospital beds used on on day _i, not new hospitalizations
        """
        # print(infections)
        # exit()
        for _i in range(region_model.N):
            start_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION-DAYS_IN_HOSPITAL)
            end_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION)

            hospitalizations[_i] = int(HOSPITALIZATION_RATE * infections[start_idx:end_idx].sum())

    ########################################
    # Compute true deaths
    ######################################## 
    # deaths_norm: the probability distribution from infected to death 
    # ifr_arr[i]: the mortality rate at day i 
    assert len(deaths_norm) % 2 == 1, 'deaths arr must be odd length'
    deaths_offset = len(deaths_norm) // 2 
    # it takes DAYS_BEFORE_DEATH days to die after being infected;  
    # print(len(deaths_norm), deaths_offset) # 15 7
    # print("DAYS_BEFORE_DEATH:", DAYS_BEFORE_DEATH)
    # exit()
    for _i in range(-deaths_offset, region_model.N-DAYS_BEFORE_DEATH):
        # we apply a convolution on the deaths norm array
        infections_subject_to_death = (infections[max(0, _i-deaths_offset):_i+deaths_offset+1] * \
            deaths_norm[:min(len(deaths_norm), deaths_offset+_i+1)]).sum()  
        true_deaths = infections_subject_to_death * region_model.ifr_arr[_i + DAYS_BEFORE_DEATH]
        deaths[_i + DAYS_BEFORE_DEATH] = true_deaths 

    ########################################
    # Compute reported deaths
    ########################################
    death_reporting_lag_arr_norm = region_model.get_reporting_delay_distribution()
    assert abs(death_reporting_lag_arr_norm.sum() - 1) < 1e-9, death_reporting_lag_arr_norm
    for i in range(region_model.N):
        """
        This section converts true deaths to reported deaths.

        We first assume that a small minority of deaths are undetected, and remove those.
        We then assume there is a reporting delay that is exponentially decreasing over time.
            The probability density function of the delay is encoded in death_reporting_lag_arr.
            In reality, reporting delays vary from region to region.
        """
        detected_deaths = deaths[i] * (1 - region_model.undetected_deaths_ratio_arr[i])
        max_idx = min(len(death_reporting_lag_arr_norm), len(deaths) - i)
        reported_deaths[i:i+max_idx] += \
            (death_reporting_lag_arr_norm * detected_deaths)[:max_idx]

    return dates, infections, hospitalizations, reported_deaths 

def compute_vaccination_rate_among_no_prev_infection(natural_infection_rate, vaccination_rate_total_population, vaccination_rate_among_prev_infected):
    if natural_infection_rate == 1:
        result = vaccination_rate_among_prev_infected
    else:
        result = (vaccination_rate_total_population - natural_infection_rate * vaccination_rate_among_prev_infected) / (1 - natural_infection_rate)
    
    result = min(1, result)
    result = max(0, result)
    
    return result