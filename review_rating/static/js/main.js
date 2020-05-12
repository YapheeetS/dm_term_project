$("button.submit-btn").on("click", function(){
    var sentence = $("textarea#game-review").val()
    console.log(sentence)

    var vote_review = (data) => {
        data_str = JSON.stringify(data)
        console.log(data_str)
        $("label.predict-title").html(`Predict rating: ${data.voting}`)
    }


    queryURL =  `http://localhost:2222/api/vote_review?text=${sentence}`
    corsHTTP(queryURL, vote_review);
})


$("button.submit-btn2").on("click", function(){
    var sentence = $("textarea#game-review2").val()
    console.log(sentence)

    var vote_review = (data) => {
        data_str = JSON.stringify(data)
        console.log(data_str)
        $("label.predict-title2").html(`Predict rating: ${data.voting}`)
    }


    queryURL =  `http://localhost:2222/api/vote_balance_review?text=${sentence}`
    corsHTTP(queryURL, vote_review);
})
