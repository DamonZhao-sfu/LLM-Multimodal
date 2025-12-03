SELECT t.ID, i.uri
FROM mmqa.ap_warrior t, mmqa.images i
WHERE LLM(
    "You will be provided with a horse racetrack name and an image. ",
    "Determine if the image shows the logo of the racetrack. ",
    "Racetrack: {text:Track} Image: {image:uri} ", t.Track, i.uri
) = 'Yes';