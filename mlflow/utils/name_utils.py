import random


def _generate_random_name(sep="-", integer_scale=3):
    """Helper function for

    :param sep: String seperator for word spacing
    :param integer_scale: dictates the maximum scale range for random integer sampling (power of 10)

    :return: A random string phrase comprised of a predicate, noun, and random integer
    """

    predicate = random.choice(_GENERATOR_PREDICATES).lower()
    noun = random.choice(_GENERATOR_NOUNS).lower()
    num = random.randint(0, 10**integer_scale)
    return f"{predicate}{sep}{noun}{sep}{num}"


_GENERATOR_NOUNS = [
    "ant",
    "ape",
    "asp",
    "auk",
    "bass",
    "bat",
    "bear",
    "bee",
    "bird",
    "boar",
    "bug",
    "calf",
    "carp",
    "cat",
    "chick",
    "chimp",
    "cod",
    "colt",
    "conch",
    "cow",
    "crab",
    "crane",
    "crocodile",
    "crow",
    "cub",
    "deer",
    "doe",
    "dog",
    "dolphin",
    "donkey",
    "dove",
    "duck",
    "eel",
    "elk",
    "fawn",
    "finch",
    "fish",
    "flea",
    "fly",
    "foal",
    "fowl",
    "fox",
    "frog",
    "gnat",
    "gnu",
    "goat",
    "goose",
    "grouse",
    "grub",
    "gull",
    "hare",
    "hawk",
    "hen",
    "hog",
    "horse",
    "hound",
    "jay",
    "kit",
    "kite",
    "koi",
    "lamb",
    "lark",
    "loon",
    "lynx",
    "mare",
    "midge",
    "mink",
    "mole",
    "moose",
    "moth",
    "mouse",
    "mule",
    "newt",
    "owl",
    "ox",
    "panda",
    "penguin",
    "perch",
    "pig",
    "pug",
    "quail",
    "ram",
    "rat",
    "ray",
    "robin",
    "roo",
    "rook",
    "seal",
    "shad",
    "shark",
    "sheep",
    "shoat",
    "shrew",
    "shrike",
    "shrimp",
    "skink",
    "skunk",
    "sloth",
    "slug",
    "smelt",
    "snail",
    "snake",
    "snipe",
    "sow",
    "sponge",
    "squid",
    "squirrel",
    "stag",
    "steed",
    "stoat",
    "stork",
    "swan",
    "tern",
    "toad",
    "trout",
    "turtle",
    "vole",
    "wasp",
    "whale",
    "wolf",
    "worm",
    "wren",
    "yak",
    "zebra",
]

_GENERATOR_PREDICATES = [
    "abundant",
    "able",
    "abrasive",
    "adorable",
    "adaptable",
    "adventurous",
    "aged",
    "agreeable",
    "ambitious",
    "amazing",
    "amusing",
    "angry",
    "auspicious",
    "awesome",
    "bald",
    "beautiful",
    "bemused",
    "bedecked",
    "big",
    "bittersweet",
    "blushing",
    "bold",
    "bouncy",
    "brawny",
    "bright",
    "burly",
    "bustling",
    "calm",
    "capable",
    "carefree",
    "capricious",
    "caring",
    "casual",
    "charming",
    "chill",
    "classy",
    "clean",
    "clumsy",
    "colorful",
    "crawling",
    "dapper",
    "debonair",
    "dashing",
    "defiant",
    "delicate",
    "delightful",
    "dazzling",
    "efficient",
    "enchanting",
    "entertaining",
    "enthusiastic",
    "exultant",
    "fearless",
    "flawless",
    "fortunate",
    "gaudy",
    "gentle",
    "gifted",
    "glamorous",
    "grandiose",
    "gregarious",
    "handsome",
    "hilarious",
    "honorable",
    "illustrious",
    "incongruous",
    "indecisive",
    "industrious",
    "intelligent",
    "inquisitive",
    "intrigued",
    "invincible",
    "judicious",
    "kindly",
    "knowledgeable",
    "languid",
    "learned",
    "legendary",
    "likeable",
    "loud",
    "luminous",
    "luxuriant",
    "lyrical",
    "magnificent",
    "marvelous",
    "masked",
    "melodic",
    "merciful",
    "mercurial",
    "monumental",
    "mysterious",
    "nebulous",
    "nervous",
    "nimble",
    "nosy",
    "omniscient",
    "orderly",
    "overjoyed",
    "peaceful",
    "painted",
    "persistent",
    "placid",
    "polite",
    "popular",
    "powerful",
    "puzzled",
    "rambunctious",
    "rare",
    "rebellious",
    "respected",
    "resilient",
    "righteous",
    "receptive",
    "redolent",
    "resilient",
    "rogue",
    "rumbling",
    "salty",
    "sassy",
    "secretive",
    "selective",
    "sedate",
    "serious",
    "shivering",
    "skillful",
    "sincere",
    "skittish",
    "silent",
    "smiling",
    "sneaky",
    "sophisticated",
    "spiffy",
    "stately",
    "suave",
    "stylish",
    "tasteful",
    "thoughtful",
    "thundering",
    "traveling",
    "treasured",
    "trusting",
    "unequaled",
    "upset",
    "unique",
    "unleashed",
    "useful",
    "upbeat",
    "unruly",
    "valuable",
    "vaunted",
    "victorious",
    "welcoming",
    "whimsical",
    "wistful",
    "wise",
    "worried",
    "youthful",
    "zealous",
]