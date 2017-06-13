def translation(original):
    # does the translation work

    #conversion to string
    original = str(original)
    # input split into words
    string_list = original.split(' ')

    # vowel list for comparison
    vowel_list = ['a', 'e', 'i', 'o', 'u', 'y']
    # final list declaration
    final_string_list = []

    for word in string_list:
        for letter in word:

            # if first letter is a vowel, add 'way' to the end, add to final string list and move on to the next word
            if word.index(letter) == 0:
                if letter.lower() in vowel_list:
                    word = word + 'way'
                    final_string_list.append(word)
                    break

            # skip consonants until a vowel is found, add the previous consonants to the end of the word, remove from the beginning, and add 'ay to the end.
            # add the translated string to the final string list
            if letter in vowel_list:
                letter_index = word.index(letter)
                word = word[letter_index:] + word[:letter_index] + 'ay'
                final_string_list.append(word)
                break
    # return the final list
    return ' '.join(final_string_list)

print("Type the phrase you want to translate:")
print(translation(input()))
