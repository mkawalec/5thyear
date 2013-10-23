matching = [x for x in range(729) if x%(3*(x/30) + 1) == 0]
print(matching, len(matching))
