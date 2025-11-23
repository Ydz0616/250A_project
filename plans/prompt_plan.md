Overview: We have a dataset of individuals’ travel itineraries described here: (Paste Markdown). 

To summarize, this tabular data contains the following columns: User Identifier, Swiss local date of the trip, Ordered index of the trip within the day, Transportation mode, Purpose at the end of the trip, and Timestamp (trip finish time in Swiss local). By treating an individual’s purpose as a time-evolving latent state sequence, which emits the observable transportation mode, our goal is to train a Hidden Markov Model (HMM) which can explain how a person’s purpose evolves across trips, and how those purposes are related to the observed modes of travel. Our goal is therefore to make the HMM capable of inferring the most likely sequence of hidden purposes (such as leisure, shopping, etc.) (Viterbi Decoding).

So far, we have processed the tabular data and have obtained the following counts (for your reference):

Transportation Modes
Mode::Car611212
Mode::Walk381088
Mode::Bicycle54471
Mode::Bus37351
Mode::LightRail 14618
Mode::Ebicycle 13236
Mode::Tram 10503
Mode::Train 9518
Mode::MotorbikeScooter 6039
Mode::RegionalTrain 5758

Trip Purposes
Home 396552
Work 214332
Leisure 162581
Shopping 109264
Wait 87729
Errand 69683
Assistance 31060
Family_friends 30223
Eat 21377
Sport 10863

YOU NEED TO COLLAPSE THE MODE AND PURPOSES, WITH THIS MAPPING:
MODE_MAPPING = {
    # Car
    "Mode::Car": "car",

    # Walk
    "Mode::Walk": "walk",

    # Bike
    "Mode::Bicycle": "bike",
    "Mode::Ebicycle": "bike",
    "Mode::MotorbikeScooter": "bike",

    # Bus
    "Mode::Bus": "bus",

    # Train (rail-based)
    "Mode::Train": "train",
    "Mode::RegionalTrain": "train",
    "Mode::LightRail": "train",
    "Mode::Tram": "train",
}



PURPOSE_MAPPING = {
    "Home": "home",
    "Work": "work",
    "Eat": "eat",

    # leisure group
    "Leisure": "leisure",
    "Sport": "leisure",
    "Family_friends": "leisure",
    "Shopping": "leisure",

    # errand group
    "Errand": "errand",
    "Assistance": "errand",
    "Wait": "errand",
}

You should also know that you can effectively ignore this column: Timestamp (trip finish time in Swiss local). Because we need to obtain a list of trips per user, along with the hidden state sequences associated with these trips, it suffices to sort first by user, then by trip date, and then by sequence index. From here, we should split our data into training, validation, and testing sets. Using the training set, we need to perform the Baum-Welch algorithm to train an HMM:

You need to help us to initialize 
1. Initial probability distribution matrix of purposes 
2. Emission probability distribution matrix of mode to purpose
3. Transition probabilities distribution matrix of purpose → purpose 

Then, validate and test the HMM using Viterbi’s, utilizing accuracy of the predicted sequence with respect to the ground truth sequence.

We will also be implementing the following baselines: 

Simple, Stupid, Rule-based:

Depending on time of day : 

it it’s 7-10 AM: predict purpose : work
If it’s 10-12 AM: predict purpose : home
If it’s 12AM-1PM: predict purpose: eat
If it’s 1PM-2PM: predict purpose: work
If it’s  2PM-5PM; predict purpose: work
If it’s 5PM-7PM : predict purpose: eat
If it’s 7PM to 10PM: predict purpose: leisure

Without using sequential data:

We have observed data, which is our training set, all mode and purposes are observed

We have testing data, which is our testing set, modes are observed but purposes are hidden 

For each mode of transportation, count the frequency of each purpose per mode, and randomly sample from those. For example, for mode “car”, if 2 people used it for work and 1 purpose used it for home, there is a ⅔ chance the purpose is work and ⅓ chance the purpose is home. Then in the testing set, weigh the chance of each purpose by the chances found in training with the observed data.




