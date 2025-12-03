SELECT t.Airlines, i.uri
FROM mmqa.tampa_international_airport t, mmqa.images i
WHERE LLM(
    "You will be provided with an airline name and an image. ",
    "Determine if the image shows the logo of the airline. ",
    "Airline: {text:Airlines} Image: {image:uri} ", t.Track, i.uri
) = 'Yes';