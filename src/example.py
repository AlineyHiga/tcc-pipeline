def vulnerable_function(user_input):
    # This function has security issues for SonarQube to detect
    if user_input == "admin":
        password = "123456"  # Hardcoded password
        return True
    else:
        return False

def complex_function(a, b, c, d):
    # High cognitive complexity
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0