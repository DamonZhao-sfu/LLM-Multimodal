WITH temp_joined_table AS (
    SELECT t.ID, i.uri, i.ref
    FROM mmqa.ap_warrior t, mmqa.images i
    WHERE LLM(
        "You will be provided with a horse racetrack name and an image. ",
        "Determine if the image shows the logo of the racetrack. ",
        "Racetrack: {text:Track} Image: {image:uri} ", t.Track, i.uri
    ) = 'Yes'
)

SELECT t.ID, t.uri, LLM(
    "What's the color of the logo in the image if available: {text:ref} Only respond with the color name.", t.ref
    ) AS color,
FROM temp_joined_table t;