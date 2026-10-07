# Each intent has 8-15 example sentences.
# The model learns word patterns from these examples.
# Add more examples to make the model smarter.

TRAINING_DATA = {
    "greeting": [
        "hello", "hi", "hey", "sup", "yo",
        "good morning", "good afternoon", "good evening",
        "howdy", "greetings", "whats up", "hey there",
        "hi how are you", "hello there", "hey buddy",
        "hiya", "hola", "heya", "good day",
        "hey how is it going", "hello friend"
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "later", "gotta go",
        "im leaving", "take care", "farewell", "peace out",
        "cya", "ttyl", "gtg", "good night", "goodnight",
        "see you later", "catch you later", "im off",
        "i have to go", "talk to you later", "bye bye"
    ],
    "thanks": [
        "thanks", "thank you", "thx", "appreciate it",
        "ty", "thank u", "many thanks", "thanks a lot",
        "cheers", "that was helpful", "thanks so much",
        "i appreciate that", "thank you very much", "big thanks"
    ],
    "identity": [
        "who are you", "what is your name", "your name",
        "what are you", "are you a bot", "are you real",
        "are you human", "tell me about yourself",
        "introduce yourself", "what kind of bot are you",
        "are you an ai", "what is your purpose"
    ],
    "creator": [
        "who made you", "who created you", "who built you",
        "your creator", "your developer", "who programmed you",
        "who coded you", "who designed you", "who wrote you"
    ],
    "status": [
        "how are you", "hows it going", "how are things",
        "how do you feel", "are you okay", "you good",
        "how are you doing", "how you been", "whats new",
        "how is everything", "how is life", "how do you do"
    ],
    "age": [
        "how old are you", "your age", "when were you born",
        "when were you made", "when were you created",
        "how long have you existed", "what year were you made"
    ],
    "joke": [
        "tell me a joke", "joke", "make me laugh",
        "something funny", "tell me something funny",
        "got any jokes", "i want to hear a joke",
        "say something funny", "tell a joke",
        "do you know any jokes", "make me smile"
    ],
    "compliment": [
        "youre cool", "youre awesome", "youre smart",
        "youre great", "i like you", "youre the best",
        "good bot", "nice bot", "you are amazing",
        "you are wonderful", "you are the best bot",
        "great job", "well done", "impressive"
    ],
    "insult": [
        "youre stupid", "you suck", "youre dumb",
        "youre useless", "youre bad", "worst bot",
        "trash bot", "youre annoying", "you are terrible",
        "you are the worst", "i hate you", "youre broken"
    ],
    "help": [
        "help", "what can you do", "commands", "options",
        "how do you work", "what do you do", "menu",
        "features", "what are your abilities", "how to use you",
        "what should i say", "give me examples"
    ],
    "feelings_sad": [
        "im sad", "im unhappy", "im depressed", "feeling down",
        "im lonely", "im upset", "im crying", "i feel terrible",
        "i had a bad day", "nothing is going right",
        "im feeling low", "everything sucks", "im miserable",
        "i feel alone", "life is hard right now"
    ],
    "feelings_happy": [
        "im happy", "im excited", "im great", "feeling good",
        "awesome day", "best day", "im glad", "im thrilled",
        "im so happy", "life is good", "everything is great",
        "im in a good mood", "feeling amazing", "im wonderful"
    ],
    "love": [
        "i love you", "do you love me", "love",
        "crush", "romance", "date me", "marry me",
        "will you be my girlfriend", "will you be my boyfriend",
        "i have feelings for you", "i adore you"
    ],
    "weather": [
        "weather", "is it raining", "is it sunny",
        "temperature", "forecast", "whats the weather",
        "is it cold outside", "is it hot today",
        "will it rain", "whats the temperature"
    ],
    "time": [
        "what time is it", "current time", "time now",
        "whats the time", "do you know the time",
        "what day is it", "what is the date"
    ],
    "food": [
        "food", "hungry", "whats for dinner", "pizza",
        "burger", "favorite food", "do you eat",
        "cooking", "what should i eat", "im starving",
        "lunch", "breakfast", "snack"
    ],
    "music": [
        "music", "song", "favorite song", "band",
        "singer", "playlist", "do you listen to music",
        "concert", "what music do you like", "spotify"
    ],
    "animals": [
        "animal", "dog", "cat", "pet", "favorite animal",
        "do you like animals", "puppy", "kitten",
        "pets", "do you have a pet"
    ],
    "programming": [
        "programming", "coding", "python", "javascript",
        "code", "developer", "software", "html", "css",
        "java", "how to code", "learn programming",
        "best programming language"
    ],
    "math": [
        "math", "mathematics", "calculate", "equation",
        "algebra", "calculus", "numbers", "what is 2 plus 2",
        "what is 2+2", "solve this", "arithmetic"
    ],
    "science": [
        "science", "physics", "chemistry", "biology",
        "experiment", "atom", "molecule", "gravity",
        "evolution", "scientific fact", "quantum"
    ],
    "space": [
        "space", "universe", "planet", "star", "moon",
        "sun", "mars", "nasa", "astronaut", "galaxy",
        "black hole", "alien", "solar system", "rocket"
    ],
    "movies": [
        "movie", "film", "cinema", "netflix", "tv show",
        "series", "watch", "favorite movie", "actor",
        "what should i watch", "recommend a movie"
    ],
    "games": [
        "game", "gaming", "video game", "play",
        "xbox", "playstation", "nintendo", "pc gaming",
        "minecraft", "fortnite", "favorite game"
    ],
    "bored": [
        "bored", "boring", "nothing to do", "im bored",
        "entertain me", "what should i do", "any suggestions",
        "i have nothing to do", "so bored right now"
    ],
    "apology": [
        "sorry", "i apologize", "my bad", "oops",
        "my mistake", "forgive me", "i didnt mean to",
        "excuse me", "pardon me"
    ],
    "agree": [
        "yeah", "yes", "yep", "yup", "sure",
        "exactly", "right", "correct", "true",
        "absolutely", "definitely", "of course", "indeed"
    ],
    "disagree": [
        "no", "nope", "nah", "wrong", "incorrect",
        "not really", "i disagree", "thats not right",
        "false", "i dont think so", "not at all"
    ],
    "confusion": [
        "what", "huh", "i dont understand", "confused",
        "what do you mean", "im confused", "explain",
        "i dont get it", "that makes no sense", "what are you saying"
    ],
    "fun_fact": [
        "fun fact", "fact", "did you know", "tell me something",
        "interesting fact", "random fact", "something interesting",
        "trivia", "tell me a fact", "i want to learn something"
    ],
    "motivation": [
        "motivate me", "inspire me", "motivation",
        "inspiration", "encouragement", "i cant do it",
        "give up", "impossible", "i need motivation",
        "i feel like quitting", "help me keep going"
    ],
    "secret": [
        "secret", "tell me a secret", "do you have secrets",
        "hidden", "easter egg", "surprise",
        "tell me something nobody knows", "confess"
    ],
    "swearing": [
        "fuck", "shit", "damn", "ass", "bitch",
        "crap", "hell", "dick", "bastard", "wtf"
    ],
    "name_tell": [
        "my name is", "call me", "i am called",
        "im called", "you can call me", "my name is john",
        "my name is sarah", "call me alex"
    ],
    "name_ask": [
        "whats my name", "do you know my name",
        "remember my name", "who am i", "what did i tell you my name was"
    ],
    "stats": [
        "stats", "statistics", "how many conversations",
        "memory", "what do you remember", "database",
        "show me stats", "conversation history"
    ],
    "teach": [
        "learn this", "teach you", "remember this",
        "new response", "add response", "i want to teach you",
        "teach me", "teach", "train you"
    ]
}