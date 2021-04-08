
        print("START TRAINING")
        for e in range(params["max_epochs"]):
            torch.cuda.empty_cache()
            for i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                loss = 0

                if params["model"] == "Standard":
                    train_y_pred = model(batch[0].to(device))
                    for task in range(model.num_tasks):
                        if params["data"] == "census":
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            task_losses[task] = criterion[task](train_y_pred[task], label.float())*params["lambda"][task]
                            loss += criterion[task](train_y_pred[task], label.float())*params["lambda"][task]
                        elif params['data'] == 'pcba':
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            loss_temp = criterion[task](train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0])*params["lambda"][task]
                            loss += loss_temp
                            if params["task_balance_method"] is not None and loss_temp < 0.0001:
                                task_losses[task] = 0.0001
                            else:
                                task_losses[task] = loss_temp
                        elif params['data'] == 'mimic':
                            #time series
                            (train_y_pred, label_inner, _, _) = maml_split(
                                batch, model, device, prop=1, time=True
                            )
                            pred, obs = organizing_predictions(
                                model, params, train_y_pred[task], label_inner, task
                            )

                            task_losses[task] = criterion[task](pred, obs).float()*params["lambda"][task]
                            loss += criterion[task](pred, obs).float()*params["lambda"][task]
                        else:
                            #your new model/dataset
                            #save the current task's labels
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            #calculate loss using the criterion function
                            loss_temp = criterion[task](train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0])*params["lambda"][task]
                            loss += loss_temp
                    loss.backward()
                    optimizer.step()
                else:
                    '''
                    MMoE or MMoEEx-diversity
                    - MMoE: Multi-gate Mixture-of-Experts
                    - Md: Multi-gate Mixture-of-Experts with Exclusivity (only diversity component, without MAML-MTl optimization)
                    '''
                    '''
                    MMoEEx: Diversity + MAML-MTL (Full model)
                    '''
                    if params["model"] == "MMoE" or params['model']== 'Md':
                        train_y_pred = model(batch[0].to(device))
                        for task in range(model.num_tasks):
                            if params["data"] == "census":
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )
                                loss += criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                            elif params['data'] == 'pcba':
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )
                                loss_temp = criterion[task](
                                    train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0]
                                )*params["lambda"][task]
                                loss += loss_temp
                                #saving loss per task for task-balancing
                                task_losses[task] = loss_temp
                            elif params['data'] == 'mimic':
                                # this maml split here is to reorganize the data nicely
                                (train_y_pred, label_inner, _, _) = maml_split(
                                    batch, model, device, prop=1, time=True
                                )
                                pred, obs = organizing_predictions(
                                    model, params, train_y_pred[task], label_inner, task
                                )
                                loss += criterion[task](pred, obs).float()*params["lambda"][task]
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](pred, obs).float()*params["lambda"][task]
                            else:
                                #your new model/dataset
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )  # .cuda()
                                loss += criterion[task](
                                    train_y_pred[task], label.float()
                                )
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                        loss.backward()
                        if params['model']=='Md':
                            #Keeping gates 'closed'
                            if params["type_exc"] == "exclusivity":
                                (
                                    model.MMoEEx.gate_kernels.grad.data,
                                    model.MMoEEx.gate_bias.grad.data,
                                ) = keep_exclusivity(model)
                            else:
                                #Exclusion
                                (
                                    model.MMoEEx.gate_kernels.grad.data,
                                    model.MMoEEx.gate_bias.grad.data,
                                ) = keep_exclusion(model)
                        optimizer.step()

                    else:
                        '''
                        1) For MAML-MTL, we split the training set in inner and outer loss calculation
                        '''
                        if params["data"] == "census":
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"]
                            )
                        elif params['data'] == 'pcba':
                            (
                                train_y_pred,
                                label_inner,
                                weight_inner,
                                train_outer,
                                label_outer,
                                weight_outer
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"], data_pcba = True
                            )
                        elif params['data'] == 'mimic':
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch,
                                model,
                                device,
                                params["maml_split_prop"],
                                True,
                                params["seqlen"],
                            )
                        else:
                            #your new model/dataset
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"]
                            )

                        loss_task_train = []
                        '''
                        2) Deepcopy to save the model before temporary updates
                        '''
                        model_copy = copy.deepcopy(model)
                        for task in range(model.num_tasks):
                            '''
                            3) Inner loss / loss in the current model
                            '''
                            pred, obs = organizing_predictions(
                                model, params, train_y_pred[task], label_inner, task
                            )
                            inner_loss = criterion[task](pred, obs).float()
                            '''
                            4) Temporary update per task
                            '''
                            params_ = gradient_update_parameters(model, inner_loss, step_size = optimizer.param_groups[0]['lr'])
                            '''
                            5) Calculate outer loss / loss in the temporary model
                            '''
                            current_y_pred = model(train_outer, params=params_)
                            pred, obs = organizing_predictions(
                                model, params, current_y_pred[task], label_outer, task,
                            )
                            loss_out = (
                                criterion[task](pred, obs).float()
                                * params["lambda"][task]
                            )
                            task_losses[task] = loss_out
                            loss += loss_out
                            loss_task_train.append(loss_out.cpu().detach().numpy())
                            '''
                            6) Reset temporary model
                            '''
                            for (_0, p_), (_1, p_b) in zip(model.named_parameters(), model_copy.named_parameters()):
                                p_.data = p_b.data

                        loss.backward()
                        #Keeping gates 'closed'
                        if params["type_exc"] == "exclusivity":
                            (
                                model.MMoEEx.gate_kernels.grad.data,
                                model.MMoEEx.gate_bias.grad.data,
                            ) = keep_exclusivity(model)
                        else:
                            (
                                model.MMoEEx.gate_kernels.grad.data,
                                model.MMoEEx.gate_bias.grad.data,
                            ) = keep_exclusion(model)
                        optimizer.step()

                """ Optional task balancing step"""
                if params["task_balance_method"] == "LBTW":
                    for task in range(model.num_tasks):
                        if i == 0:  # first batch
                            balance_tasks.get_initial_loss(
                                task_losses[task],
                                task,
                            )
                        balance_tasks.LBTW(task_losses[task], task)
                        weights = balance_tasks.get_weights()
                        params["lambda"] = weights

            if params["task_balance_method"] == "LBTW":
                print('... Current weights LBTW: ',params["lambda"])

            """ Saving losses per epoch"""
            loss_.append(loss.cpu().detach().numpy())

            print("... calculating metrics")
            if params["data"] == "census":
                print('Validation')
                auc_val, _, loss_val = metrics_census(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_census(e, train_loader, model, device, train = True)
            elif params['data']=='pcba':
                print('Validation')
                auc_val, _, loss_val = metrics_pcba(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_pcba(e, train_loader, model, device, train=True)
            elif params["data"] == "mimic":
                auc_val, loss_val, _ = metrics_mimic(e,validation_loader,model,device,params["tasks"],criterion,validation=True)
                auc_train, _, _ = metrics_mimic(e, train_loader, model, device, params["tasks"], [], training=True)
            else:
                print('Validation')
                auc_val, _, loss_val = metrics_newdata(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_newdata(e, train_loader, model, device, train = True)


            """Updating tensorboard"""
            if params["save_tensor"]:
                writer_tensorboard.add_scalar(
                    "Loss/train_Total", loss.cpu().detach().numpy(), e
                )
                for task in range(model.num_tasks):
                    writer_tensorboard.add_scalar(
                        "Auc/train_T" + str(task + 1), auc_train[task], e
                    )
                    writer_tensorboard.add_scalar(
                        "Auc/Val_T" + str(task + 1), auc_val[task], e
                    )
                writer_tensorboard.end_writer()

            """Printing Outputs """
            if e % 1 == 0:
                if params["gamma"] < 1 and e % 10 == 0 and e>1:
                    opt_scheduler.step()
                if params["rep_ci"] <= 1 and params['data'] != 'pcba':
                    print(
                        "\n{}-Loss: {} \nAUC-Train: {}  \nAUC-Val: {} \nL-val: {}".format(
                            e, loss.cpu().detach().numpy(), auc_train, auc_val, loss_val
                        )
                    )
                elif params["rep_ci"] <= 1 and params['data'] == 'pcba':
                    print(
                        "\n{}-Loss: {} \nAUC-Train-pcba: {}  \nAUC-Val: {} \nL-val: {}".format(
                            e, loss.cpu().detach().numpy(), np.nanmean(auc_train), np.nanmean(auc_val), np.mean(loss_val)
                        )
                    )

            """Saving the model with best validation AUC"""
            if params["best_validation_test"]:
                current_val_AUC = np.nansum(auc_val)
                if current_val_AUC > best_val_AUC:
                    best_epoch = e
                    best_val_AUC = current_val_AUC
                    print("better AUC ... saving model")
                    # path to save model
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["output"]
                        + "/"
                    )

                    Path(path).mkdir(parents=True, exist_ok=True)
                    path = path + "net_best.pth"
                    torch.save(model.state_dict(), path)
                print('...best epoch',best_epoch)

            """Optional: DWA task balancing"""
            if params["task_balance_method"] == "DWA":
                # add losses to history structure
                balance_tasks.add_loss_history(task_losses)
                balance_tasks.last_elements_history()
                balance_tasks.DWA(task_losses, e)
                weights = balance_tasks.get_weights()
                params["lambda"] = weights
                print('... Current weights DWA: ',params["lambda"])

            """Reset array with loss per task"""
            task_losses[:] = 0.0


        loss_ = np.array(loss_).flatten().tolist()
        torch.cuda.empty_cache()

        if params["best_validation_test"]:
            print("...Loading best validation epoch")
            path = (
                ".//output//"
                + date_start
                + "/"
                + params["model"]
                + "-"
                + params["data"]
                + "-"
                + str(params["num_experts"])
                + "-"
                + params["output"]
                + "/"
            )
            path = path + "net_best.pth"
            model.load_state_dict(torch.load(path))

        print('... calculating metrics on testing set')
        if params["data"] == "census":
            auc_test, conf_interval = metrics_census(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)
        elif params['data'] == 'pcba':
            auc_test, conf_interval = metrics_pcba(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)
            precision_auc_test = np.repeat(0,model.num_tasks)
        elif params["data"] == "mimic":
            auc_test, _, conf_interval = metrics_mimic(epoch=e,data_loader=test_loader,model=model,device=device,tasksname=params["tasks"],criterion=[],testing=True)
        else:
            auc_test, _, conf_interval = metrics_newdata(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)

        print('... calculating diversity on testing set experts')
        measuring_diversity(test_loader, model, device,params['output'],params['data'])

        """Creating and saving output files"""
        if params["rep_ci"] <= 1:
            if params['data']=='pcba':
                print("\nFinal AUC-Test: {}".format(np.nanmean(auc_test)))
            else:
                print("\nFinal AUC-Test: {}".format(auc_test))

            print("...Creating the output file")
            if params["create_outputfile"]:
                if params['data'] != 'pcba':
                    precision_auc_test = ''

                data_output = output_file_creation(
                    rep,
                    model.num_tasks,
                    auc_test,
                    auc_val,
                    auc_train,
                    conf_interval,
                    rep_start,
                    params,
                    precision_auc_test,
                )
                path = (
                    ".//output//"
                    + date_start
                    + "/"
                    + params["model"]
                    + "-"
                    + params["data"]
                    + "-"
                    + str(params["num_experts"])
                    + "-"
                    + params["task_balance_method"]
                    + "/"
                )
                data_output.to_csv(path+params['output']+'.csv', header = True,index=False)

        else:
            ci_test.append(auc_test)
            if params["create_outputfile"]:
                if rep == 0:
                    data_output = output_file_creation(
                        rep,
                        model.num_tasks,
                        auc_test,
                        auc_val,
                        auc_train,
                        conf_interval,
                        rep_start,
                        params,
                    )
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["task_balance_method"]
                        + "/"
                    )

                    data_output.to_csv(path+params['output']+'.csv', header = True,index=False)
                else:
                    _output = {"repetition": rep}
                    for i in range(model.num_tasks):
                        colname = "Task_" + str(i)
                        _output[colname + "_test"] = auc_test[i]
                        _output[colname + "_test_bs_l"] = conf_interval[i][0]
                        _output[colname + "_test_bs_u"] = conf_interval[i][1]
                        _output[colname + "_val"] = auc_val[i]
                        _output[colname + "_train"] = auc_train[i]
                    _output["time"] = time.time() - rep_start
                    _output["params"] = params
                    _output["data"] = params["data"]
                    _output["tasks"] = params["tasks"]
                    _output["model"] = params["model"]
                    _output["batch_size"] = params["batch_size"]
                    _output["max_epochs"] = params["max_epochs"]
                    _output["num_experts"] = params["num_experts"]
                    _output["num_units"] = params["num_units"]

                    _output["expert"] = try_keyerror("expert", params)
                    _output["expert_blocks"] = try_keyerror("expert_blocks", params)
                    _output["seqlen"] = try_keyerror("seqlen", params)

                    #_output["use_early_stop"] = params["use_early_stop"]
                    _output["runits"] = params["runits"]

                    _output["prop"] = params["prop"]
                    _output["lambda"] = params["lambda"]
                    _output["cw_pheno"] = try_keyerror("cw_pheno", params)
                    _output["cw_decomp"] = try_keyerror("cw_decomp", params)
                    _output["cw_ihm"] = try_keyerror("cw_ihm", params)
                    _output["cw_los"] = try_keyerror("cw_los", params)
                    _output["cw_pcba"] = try_keyerror("cw_pcba", params)
                    _output["lstm_nlayers"] = try_keyerror("lstm_nlayers", params)
                    _output["task_balance_method"] = params["task_balance_method"]
                    _output["type_exc"] = params["type_exc"]
                    _output["prob_exclusivity"] = params["prob_exclusivity"]
                    #_output["clustering"] = try_keyerror("clustering", params)
                    #_output["buckets"] = try_keyerror("buckets", params)


                    data_output = data_output.append(_output, ignore_index=True)
                    '''
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["task_balance_method"]
                        + "/"
                    )
                    data_output.to_csv(
                        path
                        + "/"
                        + datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
                        + ".csv",
                        header=True,
                        index=False,
                    )'''
                    data_output.to_csv('fall2020//output//'+params['output']+'.csv', header = True,index=False)

    """Calculating the Confidence Interval using Bootstrap"""
    if params["rep_ci"] > 1:
        model_CI(ci_test, model)
    print('...Best Epoch: ',best_epoch)
    print(params)

