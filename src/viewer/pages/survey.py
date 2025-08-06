## This is a survey page designed to collect demographic/institution data from users.
## It should be optional/skippable on local/desktop and mandatory on cloud.
import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_alerts as utils_alerts
import utils.utils_survey as utils_survey
import pycountry
import time
import re
import requests
import os

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.add_sidebar_options()
utilpg.set_global_style()

def create_survey_indicator():
    user_dir = st.session_state.paths['out_dir']
    indicator_filepath = os.path.join(user_dir, "survey.txt")
    with open(indicator_filepath, 'w') as f:
        pass

def delete_survey_indicator():
    user_dir = st.session_state.paths['out_dir']
    indicator_filepath = os.path.join(user_dir, "survey.txt")
    if os.path.exists(indicator_filepath):
        os.remove(indicator_filepath)

def clear(box):
    return lambda: box.empty()

def dummy_submit_form(form_data):
    time.sleep(2)

def submit_form(form_data):
    SUBMIT_URL = "https://7crcypuj62xqnrnmkyzaelx7ci0qrcrl.lambda-url.us-east-1.on.aws/"
    try:
        r = requests.post(SUBMIT_URL, json=form_data, timeout=5)
        if r.status_code == 200 and r.json().get("success"):
            st.success("Survey submitted successfully!")
            return True
    except Exception as e:
        st.error(f"Request failed: {e}")
        return False

def survey_panel():
    st.markdown('# NiChart User Demographics Survey')
    st.markdown('''
                NiChart is free software and a free service supported by grants that require us to ask users for demographic and institutional information.
                We ask for this so that we can measure the impact and spread of NiChart, determine how we can better serve the neuroimaging community, and meet grant milestones.
                
                You are NOT required to provide this information to use NiChart, but it would be very helpful for us. All questions are optional. To skip a question, leave it blank if it's a text field or select "Prefer not to answer".
                
                If you choose to provide this information, it will be stored by the University of Pennsylvania and accessed only for the purposes of reporting to the NIH for grant U24NS130411. All such data will be deleted upon completion of the grant period.
                If at any time you wish to view or revoke our access to the information you have provided, please contact us at software@cbica.upenn.edu and we'll be happy to help.
                ''')
    if st.session_state.has_cloud_session:
        st.markdown("Once you submit this form, you will gain **permanent, free** access to use the NiChart Cloud service and we won't ask you for this information again.")
    else: # Local
        st.markdown('''
                    When you submit this form, your responses will be sent securely to the University of Pennsylvania via the internet. Because you are running the local version of NiChart, this is the only time NiChart will connect to the internet.
                    To help us receive this data, please ensure you have an internet connection. If the transmission fails, you will still be able to use NiChart, but we will ask again at the beginning of your next session. Otherwise, we won't ask you again unless you run NiChart from a different location.
                    ''')
    
    # Identification
    name = st.text_input(label="Your name", value="")

    # Institution / Affiliation
    institution = st.text_input(label="Name of Institution", value="")

    province_select = st.empty()
    # Geographic information
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = None

    countries = ['', "Prefer not to answer"] + sorted([country.name for country in pycountry.countries])
    country_errbox = st.empty()
    selected_country = st.selectbox("Select a Country", countries, key='country_select', on_change=clear(country_errbox))
    if selected_country != st.session_state.selected_country:
        st.session_state.selected_country = selected_country

    province_errbox = st.empty()
    selected_province = 'NO PROVINCE' # Default to no province
    if selected_country and selected_country != "Prefer not to answer":
        selected_country_obj = pycountry.countries.get(name=selected_country)
        country_code = selected_country_obj.alpha_2
        subdivisions = list(pycountry.subdivisions.get(country_code=country_code))
        subdivisions = sorted([subdivision.name for subdivision in subdivisions])
    
        if subdivisions:
            selected_province = ''
            provinces = ['', "Prefer not to answer"] + subdivisions
            selected_province = st.selectbox("Select a province/state", provinces, key='province_select', on_change=clear(province_errbox))
        else:
            selected_province = 'NO PROVINCE' # Default to no province

    # Educational background info
    edu_levels = [
        '', # No answer
        "Prefer not to answer",
        "No formal education", # No ISCED classification
        "Some primary school", # ISCED 0
        "Completed primary school (elementary)", # ISCED 1
        "Completed lower secondary (middle / junior high)", # ISCED 2
        "Completed upper secondary (high school / GED)", # ISCED 3
        "Post-secondary non-tertiary (e.g. vocational training)", # ISCED 4
        "Associate's degree or equivalent", # ISCED 5
        "Bachelor's degree or equivalent", # ISCED 6
        "Master's degree or equivalent", # ISCED 7
        "Doctoral degree or equivalent (e.g. PhD, JD, MD)" # ISCED 8
    ]
    edu_errbox = st.empty()
    selected_edu = st.selectbox("Select the highest level of education completed", edu_levels, on_change=clear(edu_errbox))
    

    # Age info
    age_options = ['', "Prefer not to answer", "Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_errbox = st.empty()
    selected_age = st.selectbox("Select your age range", age_options, on_change=clear(age_errbox))
    

    # Race info
    race_options = ["Prefer not to answer",
                    "Indigenous (e.g. Native American, First Nations, Aboriginal, etc.)",
                    "Asian",
                    "Black or African descent",
                    "Middle Eastern or North African"
                    "Native Hawaiian or Other Pacific Islander",
                    "White or European descent",
                    "Other (please specify)"
                    ]
    race_errbox = st.empty()
    selected_races = st.multiselect("Please select the racial categories you identify with:",
                                   race_options,
                                   help="You may choose more than one option unless you select 'Prefer not to answer'.",
                                   on_change=clear(race_errbox))

    if "Other (please specify)" in selected_races:
        other_race_detail = st.text_input("Please specify other racial or ethnic identity:")
    else:
        other_race_detail = ''

    if "Prefer not to answer" in selected_races and len(selected_races) > 1:
        selected=["Prefer not to answer"]
    
    

    # Ethnicity Info
    ethnicity_options = ["", "Hispanic or Latino", "Not Hispanic or Latino", "Prefer not to answer"]
    ethnicity_errbox = st.empty()
    selected_ethnicity = st.selectbox("Select the ethnic category you identify with", ethnicity_options, on_change=clear(ethnicity_errbox))
    

    # Follow up email info
    st.markdown('''
                We would like to hear from you regarding your experience with NiChart. If you choose to provide your email, you may also select whether or not it is OK for us to contact you individually.
                If you select this option, we may reach out to you to ask additional questions or otherwise get your feedback on NiChart. 
                Your email will only be used to contact you for this purpose and will not be shared.
                '''
                )
    
    user_email = st.text_input("Your email address")
    BASIC_EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if user_email:
        if not re.match(BASIC_EMAIL_REGEX, user_email):
            st.error("Please enter a valid email address, or leave the field blank if you do not wish to provide it.")
    user_email_ok = st.checkbox("Yes, you may contact me for the purposes listed above.")
    
    submit = st.button("Submit Form")
    submission_msg = st.empty()
    if submit:
        any_empty_fields = False
        if len(selected_races) == 0:
            any_empty_fields = True
            race_errbox.error("You must select at least one option for race (or 'prefer not to answer').")
        if selected_ethnicity == '':
            any_empty_fields = True
            ethnicity_errbox.error("You must select an option for ethnicity (or 'prefer not to answer').")
        if selected_age == '':
            any_empty_fields = True
            age_errbox.error("You must select an option for age (or 'prefer not to answer').")
        if selected_edu == '':
            any_empty_fields = True
            edu_errbox.error("You must select an education level (or 'prefer not to answer').")
        if selected_country == '':
            any_empty_fields = True
            country_errbox.error("You must select a country (or 'prefer not to answer').")
        if selected_province == '':
            any_empty_fields = True
            province_errbox.error("You must select a state/province (or 'prefer not to answer').")
        if any_empty_fields:
            submission_msg.error('You must select an option for each field. If you do not wish to answer a question, select "Prefer not to answer", or leave the text entry blank.')
            return
        else:
            # Actually do the submission
            if st.session_state.has_cloud_session:
                user_id = st.session_state.cloud_user_id
            else:
                user_id = "LOCALUSER"

            form_data = {
                'user_id': user_id,
                'user_name': name,
                'user_email': user_email,
                'user_email_ok': str(user_email_ok),
                'user_country': selected_country,
                'user_province': selected_province,
                'user_edulevel': selected_edu,
                'user_age': selected_age,
                'user_race': ', '.join(selected_races),
                'user_ethnicity': selected_ethnicity,
                'user_race_other_detail': other_race_detail,
            }
            
            result = submit_form(form_data)

            # Create indicator file to prevent another popup
            # And add an alert to make them feel good
            if result:
                create_survey_indicator()
                utils_alerts.alert("Please accept a sincere THANK YOU from the NiChart team for filling out the user demographics survey!", type='success')
            else: # Disable the survey only for this session if the submission failed for whatever reason
                utils_alerts.alert("We couldn't submit your survey data at this time. Please check your internet connection and try again later. Until then, please enjoy NiChart!", type='warning')
                st.session_state['skip_survey'] = True
            
            
            st.switch_page("pages/home.py")
    pass


if utils_survey.is_survey_completed():
    st.markdown('''
                You are accessing the page for the NiChart user demographics survey, but it appears you have already completed it.
                If you want to make another submission, press "Take Survey". Otherwise, press "Home" to go to the home page.
                ''')
    take_survey_again = st.button("Take Survey")
    go_home = st.button("Home")
    if go_home:
        st.switch_page("pages/home.py")
    if take_survey_again:
        delete_survey_indicator()
        survey_panel()
else:
    survey_panel()

