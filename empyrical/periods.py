from enum import IntEnum

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"


class AnnualizationFactor(IntEnum):
    DAILY = APPROX_BDAYS_PER_YEAR
    WEEKLY = WEEKS_PER_YEAR
    MONTHLY = MONTHS_PER_YEAR
    QUARTERLY = QTRS_PER_YEAR
    YEARLY = 1

    def __str__(self):
        return f"{self.name.lower()}"

    @classmethod
    def periods(cls):
        return [x.name.lower() for x in cls]
