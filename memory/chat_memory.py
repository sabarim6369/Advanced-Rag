class ChatMemory:
    def __init__(self):
        self.history = []

    def add(self, user, bot):
        self.history.append(f"User: {user}\nBot: {bot}")

    def get(self):
        return "\n".join(self.history[-5:])