from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

argListPuppet = [
    {
        'name': 'accountID',
        'type': str,
        'default': None,
        'required': True,
        'help': 'Account ID in Oanda',

    },
    {
        'name': 'accessToken',
        'type': str,
        'default': None,
        'required': True,
        'help': 'Access token for specified account',

    },
    {
        'name': 'rewardStrategy',
        'type': str,
        'default': None,
        'required': True,
        'help': 'Reward strategy to be used',

    },
    {
        'name': 'instr',
        'type': str,
        'default': None,
        'required': True,
        'help': '...',

    },
    {
        'name': 'gran',
        'type': str,
        'default': None,
        'required': True,
        'help': '...',

    },
    {
        'name': 'to',
        'type': str,
        'default': None,
        'required': True,
        'help': '...',

},    {
        'name': 'from',
        'type': str,
        'default': None,
        'required': True,
        'help': '...',

    },
    {
        'name': 'numTrainIterations',
        'type': int,
        'default': None,
        'help': '...',
        'required': True,
    },
]

