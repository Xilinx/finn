/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the functions for raising events and adding them to the circular event buffer
 * for the SMBus driver.
 *
 * @file smbus_event.c
 *
 */

#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_event_buffer.h"
#include "smbus_event.h"

char* pEvent_E_TARGET_WRITE_IRQ             = "E_TARGET_WRITE_IRQ";
char* pEvent_E_TARGET_READ_IRQ              = "E_TARGET_READ_IRQ";
char* pEvent_E_TARGET_DATA_IRQ              = "E_TARGET_DATA_IRQ";
char* pEvent_E_TARGET_DONE_IRQ              = "E_TARGET_DONE_IRQ";
char* pEvent_E_TARGET_DESC_IRQ              = "E_TARGET_DESC_IRQ";
char* pEvent_E_TARGET_LOA_ERROR_IRQ         = "E_TARGET_LOA_ERROR_IRQ";
char* pEvent_E_TARGET_PEC_ERROR_IRQ         = "E_TARGET_PEC_ERROR_IRQ";
char* pEvent_E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ
                                            = "E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ";
char* pEvent_E_TARGET_RX_FIFO_ERROR_ERROR_IRQ
                                            = "E_TARGET_RX_FIFO_ERROR_ERROR_IRQ";
char* pEvent_E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ
                                            = "E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ";
char* pEvent_E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ
                                            = "E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ";
char* pEvent_E_TARGET_DESC_FIFO_ERROR_IRQ
                                            = "E_TARGET_DESC_FIFO_ERROR_IRQ";
char* pEvent_E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ
                                            = "E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ";
char* pEvent_E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ
                                            = "E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ";
char* pEvent_E_TARGET_DESC_ERROR_IRQ
                                            = "E_TARGET_DESC_ERROR_IRQ";
char* pEvent_E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ
                                            = "E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ";
char* pEvent_E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ
                                            = "E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ";
char* pEvent_E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ
                                            = "E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ";
char* pEvent_E_CONTROLLER_WRITE_IRQ         = "E_CONTROLLER_WRITE_IRQ";
char* pEvent_E_CONTROLLER_READ_IRQ          = "E_CONTROLLER_READ_IRQ";
char* pEvent_E_CONTROLLER_DATA_IRQ          = "E_CONTROLLER_DATA_IRQ";
char* pEvent_E_CONTROLLER_DONE_IRQ          = "E_CONTROLLER_DONE_IRQ";
char* pEvent_E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ
                                            = "E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ";
char* pEvent_E_CONTROLLER_LOA_ERROR_IRQ     = "E_CONTROLLER_LOA_ERROR_IRQ";
char* pEvent_E_CONTROLLER_NACK_ERROR_IRQ    = "E_CONTROLLER_NACK_ERROR_IRQ";
char* pEvent_E_CONTROLLER_PEC_ERROR_IRQ     = "E_CONTROLLER_PEC_ERROR_IRQ";
char* pEvent_E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ
                                            = "E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ";
char* pEvent_E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ
                                            = "E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ";
char* pEvent_E_CONTROLLER_RX_FIFO_ERROR_IRQ
                                            = "E_CONTROLLER_RX_FIFO_ERROR_IRQ";
char* pEvent_E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ
                                            = "E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ";
char* pEvent_E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ
                                            = "E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ";
char* pEvent_E_CONTROLLER_DESC_FIFO_ERROR_IRQ
                                            = "E_CONTROLLER_DESC_FIFO_ERROR_IRQ";
char* pEvent_E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ
                                            = "E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ";
char* pEvent_E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ
                                            = "E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ";
char* pEvent_E_CONTROLLER_DESC_ERROR_IRQ    = "E_CONTROLLER_DESC_ERROR_IRQ";
char* pEvent_E_SEND_NEXT_BYTE               = "E_SEND_NEXT_BYTE";
char* pEvent_E_IS_PEC_REQUIRED              = "E_IS_PEC_REQUIRED";
char* pEvent_E_DESC_FIFO_ALMOST_EMPTY_IRQ   = "E_DESC_FIFO_ALMOST_EMPTY_IRQ";
char* pEvent_UNKNOWN                        = "UNKNOWN";

/*******************************************************************************
*
* @brief    If the instance is valid, this function will attempt to write the event
*           into the instance's event log
*
*******************************************************************************/
void vSMBusCreateEvent( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    uint32_t ulWrite_Position = 0;
    
    if( NULL != pxSMBusInstance )
    {
        if( SMBUS_EVENT_BUFFER_FAIL == ucEventBufferTryWrite( &( pxSMBusInstance->xEventSourceCircularBuffer ), ucAnyEvent, &ulWrite_Position ) )
        {
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, 
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, ucAnyEvent, __LINE__ );
        }
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_IS_PEC_REQUIRED
*
*******************************************************************************/
void vSMBusGenerateEvent_E_IS_PEC_REQUIRED( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{   
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_IS_PEC_REQUIRED );
        /* Force the State machine to run again */
        pxSMBusInstance->ucTriggerFSM = SMBUS_TRUE;
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_SEND_NEXT_BYTE
*
*******************************************************************************/
void vSMBusGenerateEvent_E_SEND_NEXT_BYTE( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{       
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_SEND_NEXT_BYTE );
        /* Force the State machine to run again */
        pxSMBusInstance->ucTriggerFSM = SMBUS_TRUE;
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_WRITE_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_WRITE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{   
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_WRITE_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DATA_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DATA_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{   
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DATA_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DATA_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DATA_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{   
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DATA_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_READ_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_READ_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_READ_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DONE_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DONE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DONE_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DONE_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DONE_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DONE_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_LOA_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_LOA_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {   
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_LOA_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PEC_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PEC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_PEC_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_ERROR_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_ERROR_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_RX_FIFO_ERROR_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DESC_FIFO_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_DESC_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_DESC_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_LOA_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_LOA_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_LOA_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_NACK_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_NACK_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_NACK_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PEC_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PEC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_PEC_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_RX_FIFO_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DESC_FIFO_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_ERROR_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_ERROR_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DESC_ERROR_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_DESC_FIFO_ALMOST_EMPTY_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_DESC_FIFO_ALMOST_EMPTY_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_DESC_FIFO_ALMOST_EMPTY_IRQ );
    }
}

/*******************************************************************************
*
* @brief    If the instance is valid, this function will call vSMBusCreateEvent
*           with event E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ
*
*******************************************************************************/
void vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusCreateEvent( pxSMBusInstance, E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ );
    }
}

/*******************************************************************************
*
* @brief    Converts an event enum value to a text string for logging
*
*******************************************************************************/
/* Event helper functions */
char* pcEventToString( uint8_t ucEvent )
{
    char* pResult = NULL;

    switch( ucEvent )
    {
    case E_TARGET_WRITE_IRQ:
        pResult = pEvent_E_TARGET_WRITE_IRQ;
        break;

    case E_TARGET_READ_IRQ:
        pResult = pEvent_E_TARGET_READ_IRQ;
        break;

    case E_TARGET_DATA_IRQ:
        pResult = pEvent_E_TARGET_DATA_IRQ;
        break;

    case E_TARGET_DONE_IRQ:
        pResult = pEvent_E_TARGET_DONE_IRQ;
        break;

    case E_TARGET_DESC_IRQ:
        pResult = pEvent_E_TARGET_DESC_IRQ;
        break;

    case E_TARGET_LOA_ERROR_IRQ:
        pResult = pEvent_E_TARGET_LOA_ERROR_IRQ;
        break;

    case E_TARGET_PEC_ERROR_IRQ:
        pResult = pEvent_E_TARGET_PEC_ERROR_IRQ;
        break;

    case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        pResult = pEvent_E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ;
        break;

    case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        pResult = pEvent_E_TARGET_RX_FIFO_ERROR_ERROR_IRQ;
        break;

    case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        pResult = pEvent_E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ;
        break;

    case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        pResult = pEvent_E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ;
        break;

    case E_TARGET_DESC_FIFO_ERROR_IRQ:
        pResult = pEvent_E_TARGET_DESC_FIFO_ERROR_IRQ;
        break;

    case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        pResult = pEvent_E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ;
        break;

    case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        pResult = pEvent_E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ;
        break;

    case E_TARGET_DESC_ERROR_IRQ:
        pResult = pEvent_E_TARGET_DESC_ERROR_IRQ;
        break;

    case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        pResult = pEvent_E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ;
        break;

    case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        pResult = pEvent_E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ;
        break;

    case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
        pResult = pEvent_E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ;
        break;

    case E_CONTROLLER_WRITE_IRQ:
        pResult = pEvent_E_CONTROLLER_WRITE_IRQ;
        break;

    case E_CONTROLLER_READ_IRQ:
        pResult = pEvent_E_CONTROLLER_READ_IRQ;
        break;

    case E_CONTROLLER_DATA_IRQ:
        pResult = pEvent_E_CONTROLLER_DATA_IRQ;
        break;

    case E_CONTROLLER_DONE_IRQ:
        pResult = pEvent_E_CONTROLLER_DONE_IRQ;
        break;

    case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
        pResult = pEvent_E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ;
        break;

    case E_CONTROLLER_LOA_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_LOA_ERROR_IRQ;
        break;

    case E_CONTROLLER_NACK_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_NACK_ERROR_IRQ;
        break;

    case E_CONTROLLER_PEC_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_PEC_ERROR_IRQ;
        break;

    case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ;
        break;

    case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ;
        break;

    case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_RX_FIFO_ERROR_IRQ;
        break;

    case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ;
        break;

    case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ;
        break;

    case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_DESC_FIFO_ERROR_IRQ;
        break;

    case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ;
        break;

    case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ;
        break;

    case E_CONTROLLER_DESC_ERROR_IRQ:
        pResult = pEvent_E_CONTROLLER_DESC_ERROR_IRQ;
        break;

    case E_SEND_NEXT_BYTE:
        pResult = pEvent_E_SEND_NEXT_BYTE;
        break;

    case E_IS_PEC_REQUIRED:
        pResult = pEvent_E_IS_PEC_REQUIRED;
        break;

    case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
        pResult = pEvent_E_DESC_FIFO_ALMOST_EMPTY_IRQ;
        break;

    default:
        pResult = pEvent_UNKNOWN;
        break;
    }

    return pResult;
}
