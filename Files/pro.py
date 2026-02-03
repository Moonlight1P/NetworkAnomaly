# Array 'k' from the JavaScript code
k = [176, 214, 205, 246, 264, 255, 227, 237, 242, 244, 265, 270, 283]

# Username 'u' from the JavaScript code
username = "administrator"

def find_password(username, k):
    # Initialize the password as an empty string
    password = ""

    for i in range(len(username)):
        # ASCII value of the current character of the username
        u_char = ord(username[i])

        # Find the corresponding character for the password
        # Formula: p_char = k[i] - u_char - (i * 10)
        p_char = k[i] - u_char - (i * 10)

        # Convert ASCII value to character and add to the password
        password += chr(p_char)

    return password

# Call the function to guess the password
correct_password = find_password(username, k)

# Print the result
print("The correct password is:", correct_password)
