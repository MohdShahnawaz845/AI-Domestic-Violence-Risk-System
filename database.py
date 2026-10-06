import sqlite3


def create_database():

    connection = sqlite3.connect("risk_assessments.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            threats INTEGER,

            physical_harm INTEGER,

            controlling_behavior INTEGER,

            verbal_abuse INTEGER,

            isolation INTEGER,

            financial_control INTEGER,

            risk_level TEXT,

            assessment_time TEXT

        )
    """)

    connection.commit()

    connection.close()


if __name__ == "__main__":
    create_database()

    print("Database created successfully!")