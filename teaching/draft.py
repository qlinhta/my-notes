# Problem sheet 02

# Python for loop

# Exercise 9: Modify the item in the list
# A list of all european countries
european_countries = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
                      "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
                      "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
                      "Spain", "Sweden", "United Kingdom"]
# A list of 10 random countries that not necessarily are european countries
random_countries = ["China", "India", "United States", "Finland", "France", "Germany", "Greece", "Hungary", "Nigeria",
                    "Bangladesh"]

# If the country is in the list of european countries, print "This is a european country"
# If the country is not in the list of european countries, replace the country with a random country from the list of european countries"
# Print the list of countries

# Solution
for country in random_countries:
    if country in european_countries:
        print(country, "is a european country")
    else:
        random_countries[random_countries.index(country)] = european_countries[random_countries.index(country)]
print(random_countries)

# Exercise 10: We use the split method to create a list from a sentence

# Create a list of words from the sentence "The quick brown fox jumps over the lazy dog"
# Print the list of words

# Solution
sentence = "The quick brown fox jumps over the lazy dog"
list_of_words = sentence.split()
print(list_of_words)

# Exercise 11: We use the join method to create a sentence from a list of words

# Create a sentence from the list of words
# Print the sentence

# Solution
a_list_of_words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
sentence = " ".join(a_list_of_words)
print(sentence)

# Exercise 12: Create a new list with all words that are longer than 5 characters
# Solution
words = ['Long', 'words', 'include', 'grammarian', 'programmer', 'prestigious', 'and', 'beautiful']
long_words = []
for word in words:
    if len(word) > 5:
        long_words.append(word)
print(long_words)
# Other solution with comprehension list
long_words = [word for word in words if len(word) > 5]
print(long_words)

# Exercise 13: Modify the list by indexing the list
# Christmas is coming, so we want to change the list of gifts to "Christmas gifts"
'''
List of gifts: ['bike', 'car', 'house', 'boat', 'plane']
List of Christmas gifts: ['Christmas bike', 'Christmas car', 'Christmas house', 'Christmas boat', 'Christmas plane']
'''
# Solution
gifts = ['bike', 'car', 'house', 'boat', 'plane']
christmas_gifts = []
for gift in gifts:
    christmas_gifts.append("Christmas " + gift)
print(christmas_gifts)
# Other solution with comprehension list
christmas_gifts = ["Christmas " + gift for gift in gifts]
print(christmas_gifts)

