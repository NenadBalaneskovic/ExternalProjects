"""
forms.py
WTForms definitions for Weather Aggregator.
Handles user input: PO Box, City, Country.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class LocationForm(FlaskForm):
    """Form for user to enter location details."""
    po_box = StringField(
        "PO Box",
        validators=[DataRequired(), Length(min=3, max=10)],
        render_kw={"placeholder": "Enter PO Box"}
    )
    city = StringField(
        "City",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter City"}
    )
    country = StringField(
        "Country",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter Country"}
    )
    submit = SubmitField("Get Forecast")
